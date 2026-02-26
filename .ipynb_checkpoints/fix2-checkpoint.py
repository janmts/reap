with open('src/reap/observer.py', 'r') as f:
    content = f.read()

old = """                router_indices = (
                    torch.arange(batch_size * sequence_length, device=device)
                    .view(1, -1)
                    .expand(num_experts, -1)
                )
                router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
                routed_in = torch.gather(
                    input=flat_input,
                    dim=0,
                    index=router_indices,
                ).to(device)
                # we do not apply router_scores
                # record unweighted activations for all experts
                routed_out = module.experts(routed_in)
                activations = routed_out.view(num_experts, *flat_input.shape)"""

new = """                # Compute per-expert activations from fused weight tensors
                experts_module = module.experts
                gate_up_proj = experts_module.gate_up_proj  # [E, 2*I, H]
                down_proj = experts_module.down_proj  # [E, H, I]

                for expert_idx in range(num_experts):
                    expert_gate_up = gate_up_proj[expert_idx]  # [2*I, H]
                    expert_down = down_proj[expert_idx]  # [H, I]
                    gate_up_out = F.linear(flat_input.to(expert_gate_up.dtype), expert_gate_up)
                    gate_out, up_out = gate_up_out.chunk(2, dim=-1)
                    hidden = F.silu(gate_out) * up_out
                    expert_output = F.linear(hidden, expert_down)
                    activations[expert_idx] = expert_output.to(device)"""

if old in content:
    content = content.replace(old, new)
    print('Patched successfully')
else:
    print('ERROR: Could not find target text. Dumping nearby lines...')
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'routed_out = module.experts' in line:
            print(f'Found at line {i+1}')
            for j in range(max(0,i-5), min(len(lines),i+5)):
                print(f'  {j+1}: {lines[j]}')

with open('src/reap/observer.py', 'w') as f:
    f.write(content)