"""
Smoke test for the new _duration_backtest_ticker function.

Builds synthetic OHLCV data where we KNOW:
  - A 9EMA crossover happens on a specific date
  - The confirmation candle follows
  - The target will be hit within max_hold_days (WIN scenario)
  - Another trade will hit max_hold_days below entry (LOSS scenario)
  - Another trade will hit max_hold_days at breakeven (TIMEOUT scenario)

Then verifies the function returns the expected outcome classification.
"""
import sys
sys.path.insert(0, '/home/z/my-project/work')

import pandas as pd
import numpy as np
from ema9_router import _duration_backtest_ticker


def build_synthetic_df():
    """Build a synthetic OHLCV DataFrame with controlled 9EMA crossovers."""
    # 250 trading days starting 2024-01-01
    dates = pd.bdate_range('2024-01-01', periods=250)

    # Start with a slow decline (price below 9EMA) for ~60 days of warmup
    # Then create a sharp crossover + rally (WIN scenario)
    # Then decline + crossover + slow bleed (LOSS scenario)
    # Then decline + crossover + flat (TIMEOUT scenario)

    closes = []
    base = 100.0

    for i in range(250):
        if i < 60:
            # Slow decline for warmup
            base *= 0.998
            closes.append(round(base, 2))
        elif i == 60:
            # Breakout candle — gap up to trigger crossover
            closes.append(round(base * 1.05, 2))
        elif i == 61:
            # Confirmation candle — close higher
            closes.append(round(closes[-1] * 1.02, 2))
        elif 62 <= i <= 67:
            # Strong rally — target hit (entry ~107.1, target ~110.3 = +3%)
            closes.append(round(closes[-1] * 1.012, 2))
        elif 68 <= i <= 90:
            # Cool down — slow decline to set up next signal
            closes.append(round(closes[-1] * 0.997, 2))
        elif i == 91:
            # Second breakout candle
            closes.append(round(closes[-1] * 1.05, 2))
        elif i == 92:
            # Second confirmation
            closes.append(round(closes[-1] * 1.01, 2))
        elif 93 <= i <= 110:
            # Slow bleed — close will end up below entry (LOSS scenario)
            closes.append(round(closes[-1] * 0.992, 2))
        elif 111 <= i <= 130:
            # Recovery + setup for third signal
            closes.append(round(closes[-1] * 1.001, 2))
        elif i == 131:
            # Third breakout candle
            closes.append(round(closes[-1] * 1.05, 2))
        elif i == 132:
            # Third confirmation
            closes.append(round(closes[-1] * 1.01, 2))
        elif 133 <= i <= 150:
            # Flat — close stays near entry (TIMEOUT scenario, gain ~0)
            closes.append(round(closes[-1] * 1.0001, 2))
        else:
            # Fill remaining days
            closes.append(round(closes[-1] * 0.999, 2))

    df = pd.DataFrame({
        'Open':   closes,
        'High':   [c * 1.005 for c in closes],
        'Low':    [c * 0.995 for c in closes],
        'Close':  closes,
        'Volume': [1_000_000] * len(closes),
    }, index=dates)
    return df


def main():
    print('─' * 70)
    print('DURATION BACKTEST — SMOKE TEST')
    print('─' * 70)

    df = build_synthetic_df()
    print(f'\nBuilt synthetic OHLCV data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}')

    # Run backtest with date range covering the whole synthetic data
    result = _duration_backtest_ticker(
        ticker='TEST',
        df=df,
        target_pct=3.0,
        max_hold_days=15,
        require_uptrend=False,   # disable to ensure we don't filter our synthetic signals
        start_date='2024-01-01',
        end_date='2024-12-31',
    )

    print(f'\nStatus: {result["status"]}')
    print(f'Total trades: {result["summary"]["total_trades"]}')
    print(f'Wins: {result["summary"]["wins"]}')
    print(f'Losses: {result["summary"]["losses"]}')
    print(f'Timeouts: {result["summary"]["timeouts"]}')
    print(f'Win Rate: {result["summary"]["win_rate_pct"]}%')
    print(f'Total Return: {result["summary"]["total_return_pct"]}%')

    print('\nTrade details:')
    for i, t in enumerate(result['trades'], 1):
        print(f'  {i}. Entry {t["entry_date"]} @ ₹{t["entry_price"]} → '
              f'Exit {t["exit_date"]} @ ₹{t["exit_price"]} '
              f'| {t["days_held"]}d | {t["gain_pct"]:+}% | {t["outcome"]}')

    # Assertions
    s = result['summary']
    assert s['total_trades'] >= 1, 'Should have at least 1 trade'
    assert s['wins'] >= 1, 'Should have at least 1 WIN (the first rally scenario)'
    # With our synthetic data, we expect at least 1 LOSS (second scenario bleeds down)
    # and possibly a TIMEOUT (third scenario stays flat)

    outcomes = [t['outcome'] for t in result['trades']]
    print(f'\nOutcomes observed: {outcomes}')

    # Verify WIN outcome has gain_pct = +3.0 (target hit)
    wins = [t for t in result['trades'] if t['outcome'] == 'WIN']
    if wins:
        for w in wins:
            # WIN gain should be approximately +3% (target hit)
            assert abs(w['gain_pct'] - 3.0) < 0.1, \
                f'WIN trade gain_pct should be ~3.0, got {w["gain_pct"]}'
        print(f'OK: All {len(wins)} WIN trades have gain_pct ≈ +3.0% (target hit)')

    # Verify LOSS outcomes have negative gain_pct
    losses = [t for t in result['trades'] if t['outcome'] == 'LOSS']
    for l in losses:
        assert l['gain_pct'] < 0, f'LOSS trade should have negative gain_pct, got {l["gain_pct"]}'
    if losses:
        print(f'OK: All {len(losses)} LOSS trades have negative gain_pct')

    # Verify TIMEOUT outcomes have non-negative but < target gain_pct
    timeouts = [t for t in result['trades'] if t['outcome'] == 'TIMEOUT']
    for t in timeouts:
        assert -0.5 <= t['gain_pct'] < 3.0, \
            f'TIMEOUT trade should have ~0 to <3% gain, got {t["gain_pct"]}'
    if timeouts:
        print(f'OK: All {len(timeouts)} TIMEOUT trades have ~0% to <3% gain')

    print('\n' + '═' * 70)
    print('✓ ALL SMOKE TEST ASSERTIONS PASSED')
    print('═' * 70)

    # Test date filtering: narrow range should exclude some trades
    print('\nTesting date-range filtering (restricting to Mar-Apr 2024 only)...')
    result_filtered = _duration_backtest_ticker(
        ticker='TEST',
        df=df,
        target_pct=3.0,
        max_hold_days=15,
        require_uptrend=False,
        start_date='2024-03-01',
        end_date='2024-04-30',
    )
    print(f'Trades in Mar-Apr 2024 range: {result_filtered["summary"]["total_trades"]}')
    # The first signal's entry was ~2024-03-25 (day 60 = Jan 1 + 60 business days ≈ Mar 26)
    # So the filtered range should still capture it
    print('OK: Date-range filter works without error')

    # ── NEW: Test with TZ-AWARE timestamps (mimics yfinance behavior) ────────
    print('\n' + '─' * 70)
    print('Testing with TZ-AWARE timestamps (mimics yfinance Asia/Kolkata output)')
    print('─' * 70)
    df_tz = df.copy()
    df_tz.index = df_tz.index.tz_localize('Asia/Kolkata')
    print(f'DataFrame index tz: {df_tz.index.tz}')

    try:
        result_tz = _duration_backtest_ticker(
            ticker='TEST_TZ',
            df=df_tz,
            target_pct=3.0,
            max_hold_days=15,
            require_uptrend=False,
            start_date='2024-01-01',
            end_date='2024-12-31',
        )
        print(f'Status: {result_tz["status"]}')
        print(f'Total trades: {result_tz["summary"]["total_trades"]}')
        print(f'Wins: {result_tz["summary"]["wins"]} | Losses: {result_tz["summary"]["losses"]} | Timeouts: {result_tz["summary"]["timeouts"]}')
        assert result_tz["status"] == "OK", f'Should return OK status, got {result_tz["status"]}'
        assert result_tz["summary"]["total_trades"] >= 1, 'Should have at least 1 trade'
        print('OK: TZ-aware timestamps do NOT cause "Cannot compare tz-naive and tz-aware" error')
        print('OK: Trades are correctly counted with tz-aware index')

        # Verify same number of trades as tz-naive version
        result_naive = _duration_backtest_ticker(
            ticker='TEST_NAIVE',
            df=df,
            target_pct=3.0,
            max_hold_days=15,
            require_uptrend=False,
            start_date='2024-01-01',
            end_date='2024-12-31',
        )
        assert result_tz["summary"]["total_trades"] == result_naive["summary"]["total_trades"], \
            f'TZ-aware ({result_tz["summary"]["total_trades"]}) and tz-naive ({result_naive["summary"]["total_trades"]}) should give same trade count'
        print(f'OK: TZ-aware and TZ-naive give identical trade count ({result_naive["summary"]["total_trades"]})')

    except TypeError as te:
        if "tz-naive" in str(te) or "tz-aware" in str(te):
            print(f'FAIL: TZ comparison error still present: {te}')
            return 1
        raise

    print('\n' + '═' * 70)
    print('✓ ALL TESTS PASSED (including TZ-aware timestamp regression test)')
    print('═' * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
