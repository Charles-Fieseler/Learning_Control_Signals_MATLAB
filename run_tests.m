function [results, suites] = run_tests(which_tests)
% Runs some set of tests:
%   1 - sparse residual analysis functions
%   2 - CElegansModel object (base is the DMDc algorithm)
%   3 - TODO (will be sparse encoding analysis)
if ~exist('which_tests', 'var')
    which_tests = 1:2;
end

results = table();
suites = table();

warning('Turning warnings off because they are useless')
warning off
if ismember(1, which_tests)
    disp('Running tests on Sparse Residual Analysis functions')
    suite_SRA = matlab.unittest.TestSuite.fromFolder(...
        './tests/Sparse_residual_analysis_tests');
    result_SRA = run(suite_SRA);
    results = my_append(results, {result_SRA});
    suites = my_append(suites, {suite_SRA});
end
if ismember(2, which_tests)
    disp('Running tests on CElegansModel object and functions')
    suite_CEM = matlab.unittest.TestSuite.fromFolder(...
        './tests/CElegansModel_tests');
    result_CEM = run(suite_CEM);
    results = my_append(results, {result_CEM});
    suites = my_append(suites, {suite_CEM});
end

warning on
end