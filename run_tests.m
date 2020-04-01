function [results, suites] = run_tests(which_tests)
% Runs some set of tests
if ~exist('which_tests', 'var')
    which_tests = 1:2;
end

results = table();
suites = table();

warning('Turning warnings off because they are useless')
warning off
if ismember(1, which_tests)
    disp('Running tests on util functions')
    suite_util= matlab.unittest.TestSuite.fromFolder(...
        './tests/util');
    result_util = run(suite_util);
    results = my_append(results, {result_util});
    suites = my_append(suites, {suite_util});
end
% if ismember(2, which_tests)
% %     disp('Running tests on CElegansModel object and functions')
%     suite_CEM = matlab.unittest.TestSuite.fromFolder(...
%         './tests/CElegansModel_tests');
%     result_CEM = run(suite_CEM);
%     results = my_append(results, {result_CEM});
%     suites = my_append(suites, {suite_CEM});
% end

warning on
end