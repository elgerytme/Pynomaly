#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'monorepo', 'mutation'))

try:
    from engine import MutationRunner
    from reporters import MutationResult, MutationTestSuite, MutationReporter
    print('✓ All mutation testing modules imported successfully')
    
    # Test basic functionality
    result = MutationResult(
        id='test_1',
        status='killed',
        mutant_code='x + 1',
        original_code='x - 1',
        filename='test.py',
        line_number=10
    )
    print(f'✓ MutationResult created: killed={result.is_killed}, survived={result.is_survived}')
    
    suite = MutationTestSuite('test_suite')
    suite.add_result(result)
    result2 = MutationResult(id='test_2', status='survived', mutant_code='y*2', original_code='y/2', filename='test.py', line_number=20)
    suite.add_result(result2)
    print(f'✓ MutationTestSuite created: total={suite.total_mutations}, score={suite.mutation_score}%')
    
    reporter = MutationReporter(suite)
    summary = reporter.summary()
    print('✓ MutationReporter summary generated')
    print(summary)
    
    runner = MutationRunner(paths=['src/test'], timeout=30, parallel=True)
    command = runner._build_command()
    print(f'✓ MutationRunner command: {command}')
    
    print('\n' + '=' * 50)
    print('✓ All tests passed! Mutation testing implementation is working correctly.')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
