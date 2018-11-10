# ***Only need the LAST checkpoint folder - otherwise, merging the data
# will take approximately forever***


from dedalus.tools import post
# post.merge_process_files("slices", cleanup=True)
# post.merge_process_files("dump",cleanup=True)
post.merge_process_files("checkpoint",cleanup=True)

import subprocess
# print(subprocess.check_output("find slices", shell=True).decode())
# print(subprocess.check_output("find dump", shell=True).decode())
print(subprocess.check_output("find checkpoint", shell=True).decode())



import pathlib
# set_paths=list(pathlib.Path("slices").glob("slices_s*.h5"))
# post.merge_sets("slices/slices.h5",set_paths,cleanup=True)
# set_paths=list(pathlib.Path("dump").glob("dump_s*.h5"))
# post.merge_sets("dump/dump.h5",set_paths,cleanup=True)
set_paths=list(pathlib.Path("checkpoint").glob("checkpoint_s*.h5"))
post.merge_sets("checkpoint/checkpoint.h5",set_paths,cleanup=True)



# Rename folders
import os
# os.rename("dump", "dump_old")
# os.rename("slices", "slices_old")
os.rename("checkpoint", "checkpoint_old")
