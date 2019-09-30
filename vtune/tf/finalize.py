#!/usr/bin/env python
import sys,os
string = 'amplxe: Warning: Cannot locate file `'
output = ''
for line in open(sys.argv[1]):
   if string in line:
      filename = line[len(string):-3]
      if os.path.exists(filename):
         output += ' -search-dir ' + os.path.dirname(filename)
print(output)
