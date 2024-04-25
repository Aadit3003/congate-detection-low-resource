def file_read_strings(path):
    """
    Read a file into a list of strings. If the file cannot be
    read, print an error message and return an empty list.
    """
    try:
        f = open (path, 'rb')
        contents = f.read().decode("latin-").splitlines()
        f.close ()
        return contents
    except Exception as e:
        print(f'Error: Cannot read {path}\n    {str(e)}')
        return None

def file_write_strings(path, lst):
    """
    Write a list of strings (or things that can be converted to
    strings) to a file. If the file cannot be written, print an
    error message.

    path: A file path
    lst: A list of strings or things that can be converted to strings.
    """
    try:
        f = open (path, 'w')
        for l in lst:
            f.write(str(l) + '\n')
    except Exception as e:
        print(f'Error: Cannot write {path}\n    {str(e)}')
        return None