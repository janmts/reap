python -c "
import os
for root, dirs, files in os.walk('src/reap'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'r') as fh:
                content = fh.read()
            if 'vllm' in content:
                print(f'Patching {path}')
                content = content.replace(
                    'from vllm import TokensPrompt',
                    'try:\n    from vllm import TokensPrompt\nexcept ImportError:\n    TokensPrompt = None'
                )
                content = content.replace(
                    'from vllm.entrypoints', '# from vllm.entrypoints'
                )
                content = content.replace(
                    'from vllm import ', '# from vllm import '
                )
                with open(path, 'w') as fh:
                    fh.write(content)
print('Done')
"
