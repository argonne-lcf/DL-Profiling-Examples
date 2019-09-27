kernprof -l  mnist.py


# look at the results with
# $ python -m line_profiler mnist.py.lprof
# Timer unit: 1e-06 s
#
# Total time: 33.7614 s
# File: mnist.py
# Function: forward at line 30
#
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     30                                               @profile
#     31                                               def forward(self, x):
#     32       938    6534393.0   6966.3     19.4        x = self.conv1(x)
#     33       938    2792883.0   2977.5      8.3        x = torch.relu(x)
#     34
#     35       938   11571707.0  12336.6     34.3        x = self.pool1(x)
#     36
#     37       938   10071493.0  10737.2     29.8        x = self.conv2(x)
#     38       938     826315.0    880.9      2.4        x = torch.relu(x)
#     39
#     40                                                 # Flatten will flatten arbitrary tensors, so specify from the channel index to the end:
#     41       938      18347.0     19.6      0.1        x = torch.flatten(x, 1, -1)
#     42       938    1735272.0   1850.0      5.1        x = self.d1(x)
#     43       938      28303.0     30.2      0.1        x = torch.relu(x)
#     44       938     114232.0    121.8      0.3        x = self.dropout(x)
#     45       938      67717.0     72.2      0.2        x = self.d2(x)
#     46       938        746.0      0.8      0.0        return x
#
#
