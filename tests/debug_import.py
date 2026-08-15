print("Script started")
import sys

print("Imported sys")
import os

print("Imported os")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("Added to path")


print("Imported assemble_context")


print("Imported models")

print("All imports successful!")
