import ssl
from geopy.geocoders import Nominatim

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

geolocator = Nominatim(user_agent="tsp_solver", ssl_context=ctx)