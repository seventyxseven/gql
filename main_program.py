# Import hello module
import hello
import add
import subtract


import importlib
importlib.reload(subtract)

# Now calls the new version of the function

# Call function
hello.world()
print(add.add(2,1))
print(subtract.subtract(2,1))
