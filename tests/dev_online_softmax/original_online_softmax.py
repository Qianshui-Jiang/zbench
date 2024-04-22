import torch

torch.manual_seed(456)


row_count, col_count = 4, 16
long_input_vec: torch.Tensor = torch.rand((row_count, col_count))



# torch softmax as a reference
expected_softmax = torch.softmax(long_input_vec, dim=1)

# 1st read, torch max output both indexes and values, we only want the values
# we transpose it to get a vertical tensor
row_max = torch.max(long_input_vec, dim=1).values[:, None]
print("input row max\n", row_max)
# 2nd read
input_safe = long_input_vec - row_max
print("Below we reduce values amplitude, that's the safe part of safe softmax")
print("original 1st row input:\n", long_input_vec[0, :], "\nsafe softmax input 1st row:\n", input_safe[0, :])

softmax_numerator = torch.exp(input_safe)
# 3rd read
normalizer_term = torch.sum(softmax_numerator, dim=1)[:, None]
# 4th read
naive_softmax = softmax_numerator / normalizer_term

assert torch.allclose(naive_softmax, expected_softmax)
print("------------PASS-1------------")


# online softmax
online_softmax = torch.zeros_like(long_input_vec)

for row in range(row_count):
    row_max = 0.0
    normalizer_term = 0.0
    print("--- new row ---")
    for col in range(col_count):  # scalar level iteration
        val = long_input_vec[row, col]
        old_row_max = row_max
        row_max = max(old_row_max, val)
        # np.exp(old_max_row - max_row) is the adjustment factor of our precedent normalizer term,
        # after this multiplication it's like we had always substracted row_max up to this point
        # instead of old_row_max
        normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(val - row_max)
        if old_row_max != row_max:
            print("new max discovered")
        print(f"current row max: {row_max}, denominator: {normalizer_term}")

    # leverage our 2 statistics
    online_softmax[row, :] = torch.exp(long_input_vec[row, :] - row_max) / normalizer_term

assert torch.allclose(online_softmax, expected_softmax)
print("------------PASS-2------------")
