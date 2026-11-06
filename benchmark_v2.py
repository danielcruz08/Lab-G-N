"""Standalone benchmark WLS; requires NumPy and preprocessed course CSVs.
All clock/atmosphere/TGD corrections are metres. No reference truth is read.
Default data paths assume this script is kept in the gnss_spp folder.
"""
import csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent

def correct_pseudoranges(data):
    return data['pseudorange'] + data['satellite_clock'] - data['ionosphere'] - data['troposphere']

def build_geometry_matrix(satellite_positions,
                          corrected_pseudoranges,
                          receiver_state):

    receiver_position = receiver_state[:3]
    clock_bias = receiver_state[3]

    delta = satellite_positions - receiver_position

    ranges = np.linalg.norm(delta, axis=1)

    if np.any(ranges <= 0):
        raise ValueError("Invalid geometric distance")

    H = np.column_stack((
        -delta / ranges[:, None],
        np.ones(len(ranges))
    ))

    predicted = ranges + clock_bias

    residuals = corrected_pseudoranges - predicted

    return H, residuals


def solve_spp(satellite_positions, corrected_pseudoranges,
              max_iterations=30, tolerance=1e-4):

    state = np.zeros(4)

    if len(corrected_pseudoranges) < 4:
        raise ValueError("At least four satellites required")

    for iteration in range(max_iterations):

        H, residuals = build_geometry_matrix(
            satellite_positions,
            corrected_pseudoranges,
            state
        )

        # Use equal weights until a plausible position exists
        if np.linalg.norm(state[:3]) > 6e6:
            weights = calculate_weights(
                satellite_positions,
                state[:3]
            )
        else:
            weights = np.ones(len(residuals))

        sqrt_w = np.sqrt(weights)

        H_weighted = H * sqrt_w[:, None]
        y_weighted = residuals * sqrt_w

        correction, _, rank, _ = np.linalg.lstsq(
            H_weighted,
            y_weighted,
            rcond=None
        )

        if rank < 4:
            raise ValueError("Insufficient satellite geometry")

        state += correction

        if np.linalg.norm(correction) < tolerance:
            return state, iteration + 1

    raise RuntimeError("SPP did not converge")




def calculate_weights(satellite_positions, receiver_position):
    """
    Calculate elevation-based weights for GNSS observations.
    """

    # Approximate local Up direction
    up = receiver_position / np.linalg.norm(receiver_position)

    # Receiver-to-satellite vectors
    delta = satellite_positions - receiver_position

    ranges = np.linalg.norm(delta, axis=1)

    # Sine of satellite elevation
    sin_elevation = (delta @ up) / ranges

    # Elevation-based weights
    weights = np.maximum(sin_elevation, 0.05)**2

    return weights





HEADER = ['gps_week', 'gps_tow_s', 'x_ecef_m', 'y_ecef_m', 'z_ecef_m']
NAMES = dict(pseudorange='pseudoranges_meas', satellite_position='satellite_positions',
             satellite_clock='satellite_clock_bias', ionosphere='ionospheric_delay',
             troposphere='tropospheric_delay', group_delay='group_delay')
ORIGINAL_GEOMETRY = build_geometry_matrix


def gps_geometry(sp, pr, state):
    H, residual = ORIGINAL_GEOMETRY(sp, pr, state)
    # RTKLIB geodist first-order Earth rotation range term. sp is transmission ECEF.
    k = 7.2921151467e-5 / 299792458.0
    rotation = k * (sp[:, 0]*state[1] - sp[:, 1]*state[0])
    H[:, 0] -= k*sp[:, 1]
    H[:, 1] += k*sp[:, 0]
    return H, residual-rotation


def load(folder, gps=True):
    data = {key: np.loadtxt(folder/(name+'.csv'), delimiter=',', ndmin=2)
            for key, name in NAMES.items() if gps or key != 'group_delay'}
    m, n = data['pseudorange'].shape
    for key, value in data.items():
        expected = (m, 3*n) if key == 'satellite_position' else (m, n)
        if value.shape != expected:
            raise ValueError(f'{key}: {value.shape} != {expected}')
    with (folder/'epochs.csv').open(encoding='utf-8-sig', newline='') as f:
        epochs = [(int(r['gps_week']), float(r['gps_tow_s'])) for r in csv.DictReader(f)]
    keys = [(w, round(t, 1)) for w, t in epochs]
    if len(epochs) != n or keys != sorted(set(keys)):
        raise ValueError('Epoch count/order/uniqueness mismatch')
    if any(w < 0 or not np.isfinite(t) or not 0 <= t < 604800 for w, t in epochs):
        raise ValueError('Invalid GPST')
    return data, epochs


def solve(data, epochs, gps=True):
    global build_geometry_matrix
    build_geometry_matrix = gps_geometry if gps else ORIGINAL_GEOMETRY
    corrected = correct_pseudoranges(data)
    if gps:
        corrected = corrected-data['group_delay']
    rows, details, omitted = [], [], []
    for i, (week, tow) in enumerate(epochs):
        sp = data['satellite_position'][:, 3*i:3*i+3]
        pr = corrected[:, i]
        use = np.isfinite(sp).all(axis=1) & np.isfinite(pr) & (pr > 1e7)
        count = int(use.sum())
        try:
            state, iterations = solve_spp(sp[use], pr[use])
            if not np.isfinite(state).all() or not 6.30e6 < np.linalg.norm(state[:3]) < 6.42e6:
                raise ValueError('Invalid ECEF surface position')
            H, residual = build_geometry_matrix(sp[use], pr[use], state)
            pdop = float(np.sqrt(np.trace(np.linalg.inv(H.T@H)[:3, :3])))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            omitted.append([week, f'{tow:.6f}', count, str(exc)])
            continue
        rows.append([week, f'{tow:.1f}', *[f'{v:.3f}' for v in state[:3]]])
        details.append([i, week, tow, *state, count, iterations,
                        float(np.sqrt(np.mean(residual**2))),
                        float(np.max(np.abs(residual))), pdop])
    return rows, details, omitted


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)



def main():
    data15, epochs = load(ROOT/'benchmark/preprocessed_gps')
    data10, epochs10 = load(ROOT/'experiments/el10/inputs')
    if epochs != epochs10:
        raise ValueError('Input epoch grids differ')
    rows15, details15, _ = solve(data15, epochs)
    rows10, details10, _ = solve(data10, epochs)
    by_epoch = {int(d[0]): row for d, row in zip(details15, rows15)}
    added = 0
    for d, row in zip(details10, rows10):
        if int(d[0]) not in by_epoch:
            by_epoch[int(d[0])] = row
            added += 1
    rows = [by_epoch[i] for i in sorted(by_epoch)]
    write_csv(ROOT/'benchmark_v2.csv', HEADER, rows)
    print('Solved:', len(rows), '/', len(epochs))
    print('Additional epochs:', added)
    print('Availability (%):', 100*len(rows)/len(epochs))

if __name__ == '__main__':
    main()
