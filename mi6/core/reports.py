# Module for Logging & Printing Relevant Information




# Color Printing
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


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman. (a very cool person that knows a thing or two about Policy Optimization)
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


# Main Util to print output nice (cp = color print)
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

    assert item != None, "No item passed to 'reports.cp()' to print"
    if color == None:
        print(item)
    else:
        item = colorize(item, color, bold, highlight)
        print(item)


