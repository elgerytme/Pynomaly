
import os
import subprocess

# Potential security issues for testing
password = "hardcoded_password"
user_input = "test_user"
user_command = "echo hello"
shell_command = "echo test"
sql_query = "SELECT * FROM users WHERE name = '%s'" % user_input
os.system(user_command)
subprocess.call(shell_command, shell=True)
