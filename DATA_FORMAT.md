# Test-data format

Each preprocessed scene is one NumPy .npy file containing a numeric array of
shape N x 7 (additional columns are ignored):

| Column | Meaning | Expected values |
|---:|---|---|
| 0 | x coordinate | floating point |
| 1 | y coordinate | floating point |
| 2 | z coordinate | floating point |
| 3 | red channel | 0-255 |
| 4 | green channel | 0-255 |
| 5 | blue channel | 0-255 |
| 6 | semantic label | integer 0, 1, or 2 |

Label mapping:

| Label | Class |
|---:|---|
| 0 | trunk |
| 1 | leaf |
| 2 | ground |

For the default --test-area 3, filenames must contain the token Area_3, for
example:

~~~text
data/stanford_indoor3d/Area_3_scene_01.npy
data/stanford_indoor3d/Area_3_scene_02.npy
~~~

The loader rejects object arrays, empty scenes, non-finite values, non-integer
labels, and labels outside [0, 2]. During evaluation, each scene is divided
into overlapping 1 m x 1 m XY blocks with 0.5 m stride by default. RGB values
are divided by 255, and three scene-normalized XYZ channels are appended,
producing the nine input channels expected by the released checkpoint.