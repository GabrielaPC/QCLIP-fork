from collections import Counter

def einsum2interleaved(expr):
    if isinstance(expr, str):
        if '->' not in expr:
            raise ValueError("Invalid einsum string format. Missing '->' operator.")
            
        lhs, rhs = expr.split('->')
        input_indices = [list(tok.strip()) for tok in lhs.split(',') if tok.strip()]
        out_list = list(rhs.strip())
        
        return [input_indices, out_list]
    
def interleaved2einsum(input_indices, out_list):
    lhs = ",".join("".join(tensor) for tensor in input_indices)
    rhs = "".join(out_list)
    
    return f"{lhs}->{rhs}"

def sort_tn(tn):
    remaining = list(tn)
    sorted_tn = []

    counts = Counter(sum([tensor[1] for tensor in tn], []))
    boundary_indices = {k for k, v in counts.items() if v == 1}

    for tensor in list(remaining):
        if any(idx in boundary_indices for idx in tensor[1]):
            sorted_tn.append(tensor)
            remaining.remove(tensor)
    
    current_indices = set()
    for tensor in sorted_tn:
        current_indices.update(tensor[1])

    while remaining:
        for tensor in list(remaining):
            if any(idx in current_indices for idx in tensor[1]):
                sorted_tn.append(tensor)
                remaining.remove(tensor)
                current_indices.update(tensor[1])
                break

    return list(reversed(sorted_tn))

def modal_compose(text_einsum, text_tensor_arr, img_einsum, img_tensor_arr):
    text_indices = text_einsum[0]
    all_text_chars = {char for tensor in text_indices for char in tensor}
    max_text_char = max(all_text_chars) if all_text_chars else 'a'
    char_offset = ord(max_text_char) + 1

    text_outputs = text_einsum[1]
    img_init_positions = [i for i, tensor in enumerate(img_tensor_arr) if tensor == (None, '0')]
    if len(img_init_positions) != len(text_outputs):
        raise ValueError("Dimension Mismatch between Text outputs and Image inputs!")
    
    img_indices = img_einsum[0]
    boundary_wire_map = {img_indices[pos][0]: text_outputs[i] for i, pos in enumerate(img_init_positions)}

    cleaned_img_indices = []
    cleaned_img_tensor_arr = []
    init_set = set(img_init_positions)

    for i, tensor in enumerate(img_indices):
        if i in init_set:
            continue
        remapped_tensor = []
        for wire_char in tensor:
            if wire_char in boundary_wire_map:
                remapped_tensor.append(boundary_wire_map[wire_char])
            else:
                new_char = "".join(chr(ord(c) + char_offset) for c in wire_char)
                remapped_tensor.append(new_char)
        cleaned_img_indices.append(remapped_tensor)
        cleaned_img_tensor_arr.append(img_tensor_arr[i])
    unified_indices = text_indices + cleaned_img_indices
    unified_tensor_arr = text_tensor_arr + cleaned_img_tensor_arr

    final_outputs = []
    for wire_char in img_indices[-1]:
        if wire_char in boundary_wire_map:
            final_outputs.append(boundary_wire_map[wire_char])
        else:
            new_char = "".join(chr(ord(c) + char_offset) for c in wire_char)
            final_outputs.append(new_char)
    return (unified_indices, final_outputs), unified_tensor_arr