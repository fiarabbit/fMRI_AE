import nibabel as nib
m = nib.load("average_optthr.nii").get_data()
s = m.shape

starts = [-1, -1, -1]
ends = [-1, -1, -1]
for x in range(s[0]):
    if m[x, :, :].any():
        if starts[0] == -1:
            starts[0] = x
        ends[0] = x

for y in range(s[1]):
    if m[:, y, :].any():
        if starts[1] == -1:
            starts[1] = y
        ends[1] = y

for z in range(s[2]):
    if m[:, :, z].any():
        if starts[2] == -1:
            starts[2] = z
        ends[2] = z

assert starts == [10, 11, 3]
assert ends == [79, 98, 76]

m = nib.load("binary_mask4grey_BerlinMargulies26subjects.nii").get_data()
s = m.shape

starts = [-1, -1, -1]
ends = [-1, -1, -1]
for x in range(s[0]):
    if m[x, :, :].any():
        if starts[0] == -1:
            starts[0] = x
        ends[0] = x

for y in range(s[1]):
    if m[:, y, :].any():
        if starts[1] == -1:
            starts[1] = y
        ends[1] = y

for z in range(s[2]):
    if m[:, :, z].any():
        if starts[2] == -1:
            starts[2] = z
        ends[2] = z

assert starts == [10, 10, 3]
assert ends == [79, 99, 76]