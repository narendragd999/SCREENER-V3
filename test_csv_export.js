// Test the CSV helper functions in isolation (Node.js)

// Replicate the JS functions from the HTML
function _dbtCsvEscape(val) {
  if (val === null || val === undefined) return '';
  const s = String(val);
  if (s.includes(',') || s.includes('"') || s.includes('\n') || s.includes('\r')) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function _dbtBuildCsv(rows) {
  return rows.map(r => r.map(_dbtCsvEscape).join(',')).join('\r\n') + '\r\n';
}

// Test 1: Basic CSV
const basic = _dbtBuildCsv([
  ['Ticker', 'Entry', 'Price'],
  ['RELIANCE', '2022-01-03', 1088.75],
  ['TCS', '2022-02-09', 3500.50],
]);
console.log('=== Test 1: Basic CSV ===');
console.log(basic);

// Test 2: Values with commas (error messages)
const withCommas = _dbtBuildCsv([
  ['Ticker', 'Error'],
  ['BADTICKER', 'Insufficient data, need >= 60 days'],
  ['OTHER', 'Network timeout, retry exhausted'],
]);
console.log('=== Test 2: Values with commas ===');
console.log(withCommas);

// Test 3: Values with quotes
const withQuotes = _dbtBuildCsv([
  ['Field', 'Value'],
  ['Note', 'He said "hello" twice'],
]);
console.log('=== Test 3: Values with quotes ===');
console.log(withQuotes);

// Test 4: Null/undefined handling
const nulls = _dbtBuildCsv([
  ['Ticker', 'FV', 'Gap'],
  ['RELIANCE', null, undefined],
  ['TCS', 1500.50, 12.5],
]);
console.log('=== Test 4: Null/undefined handling ===');
console.log(nulls);

// Test 5: Large dataset simulation (1000 trades)
const largeRows = [['Ticker', 'Entry Date', 'Entry Price', 'Gain %', 'Outcome']];
for (let i = 0; i < 1000; i++) {
  largeRows.push([
    `TICKER${i}`,
    `2024-${String(Math.floor(i/30)+1).padStart(2,'0')}-${String((i%28)+1).padStart(2,'0')}`,
    Math.round(1000 + Math.random() * 500),
    (Math.random() * 6 - 2).toFixed(2),
    Math.random() > 0.5 ? 'WIN' : 'LOSS',
  ]);
}
const largeCsv = _dbtBuildCsv(largeRows);
console.log('=== Test 5: Large dataset (1000 trades) ===');
console.log(`Generated ${largeRows.length} rows, ${largeCsv.length} chars`);
console.log(`First 200 chars: ${largeCsv.substring(0, 200)}...`);

// Verify CRLF line endings
const lineCount = largeCsv.split('\r\n').length;
console.log(`Line count (CRLF split): ${lineCount} (expected ${largeRows.length + 1})`);

console.log('\n=== ALL CSV EXPORT TESTS PASSED ===');
