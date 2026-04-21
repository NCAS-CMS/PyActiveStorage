import p5rem
import fsspec
import cf


def print_handle_state(label, handle):
    closed = getattr(handle, "closed", None)
    if closed is None:
        print(f"{label}: handle state unavailable")
        return

    print(f"{label}: open={not closed}, closed={closed}")

session = p5rem.bootstrap_session(
    host="xfer1",
    login_shell=True,
    use_cache=False,
    remote_python="conda run -n jas26 python",
)

fs = fsspec.filesystem("https")
http_file = 'https://gws-access.jasmin.ac.uk/public/canari/bnl/test1.nc'

print ('Opening remote file via HTTPS...')

try: 
    remote_file = fs.open(http_file)
    fields = cf.read(remote_file)
    field = fields[0]       

    print(field)

    print_handle_state("HTTPS file after print(field)", remote_file)

    

    for key, coord in field.dimension_coordinates(todict=True).items():
        print(key, coord.identity(default="unknown"), coord.array.shape)    

finally:
    remote_file.close() 


print('Opening remote file via SSH...')

try:
    remote_file = session.open("canari/public/bnl/test1.nc")
    fields = cf.read(remote_file)
    field = fields[0]

    print(field)

    print_handle_state("SSH file after print(field)", remote_file)

    for key, coord in field.dimension_coordinates(todict=True).items():
        print(key, coord.identity(default="unknown"), coord.array.shape)

    print(field.array.shape)
finally:
    session.close()







