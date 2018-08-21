% Runs all tests in the folder
warning('Turning warnings off because they are useless')
warning off
suite = matlab.unittest.TestSuite.fromFolder('./tests/');
result = run(suite);
warning on