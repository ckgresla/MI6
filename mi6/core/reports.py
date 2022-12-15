# Module for Printing Information with Style


# Map Color Names to Terminal Ints -- defaults to the terminal colors (like your own colorscheme)
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=39,
    #additional colors do string highlighting (i.e, 42 prints strings highlighted green w white text) or are plain white
    red_highlight=41,
    green_highlight=42,
    yellow_highlight=43,
    blue_highlight=44,
)


# Adds color/formatting to the provided string -- result from this can be printed out with the magic
def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman, the policy optimization guy
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


# Main Util to print output given a formatting specification (cp = color print)
def cp(item, color=None, bold=False, highlight=False):
    """
    item : stdout, this is the thing you want to print (typically any STDOUT should do)
    color : string, of one of the supported colors (will print out the item with this color instead of the default STDOUT)
    bold : bool, flag to make the output Bold or not (default False)
    highlight : bool, flag to higlight the output or not (default False)

    Supported colors are:
    [gray, red, green, yellow, blue, magenta, cyan, white, crimson]
      *the colors above will change from terminal to terminal, as they are related to a User's current color scheme
    """

    assert item != None, "No item passed to 'cp()' to print"
    if color == None:
        print(item)
    else:
        item = colorize(item, color, bold, highlight)
        print(item)


# Quick Utils for [Errors, Warnings & Logs/Info]
def printerr(item):
    cp(item, "red")

def printwar(item):
    cp(item, "yellow")

def printlog(item):
    cp(item, "blue")




# Testing out Printing File of Calling (for success/fail color coded prints in import statements) for debugging
# import inspect

# def file_executing(idx=2):
#     """
#     Prints the File that calls this function --> used to debug Document Store Connections @ brain
#       - should print something like 'path/to/file:line_number'
#     as per- https://stackoverflow.com/questions/24438976/debugging-get-filename-and-line-number-from-which-a-function-is-called
#     """
#     # caller = inspect.getframeinfo(inspect.stack()[2][0]) #when connection is called (stack trace can get kinda complex)
#     # Configure Index of File info to include based on Docker or Local File
#     if RUNNING_IN_DOCKER:
#         stack_caller_idx = idx #should be "2", typically is file of interest
#     else:
#         stack_caller_idx = idx #last index works mainly for one-off files
#     caller = inspect.stack()[stack_caller_idx]

#     debug_str = f"{caller.filename}:{caller.lineno}" #something like; "/path/to/file:line_number"
#     return debug_str


