import math
import os

# Generate location data
myaddress = "Cory Hall Berkeley CA USA"

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="EE123")
location = geolocator.geocode(myaddress)

print(location.address)
print((location.latitude, location.longitude))

# record results
lat = "lat=%d^%2.2f%s" % (int(abs(location.latitude)),
                          60*(abs(location.latitude) - abs(math.floor(location.latitude))),
                          ("N") if location.latitude >0 else "S" )  
long = "long=%d^%2.2f%s" % (int(abs(location.longitude)),
                            60*(abs(location.longitude) - math.floor(abs(location.longitude))),
                          ("W") if location.longitude <0 else "E" )  
                            
print(lat, long)

# enter your callsign and comment for the beacon
callsign = "EE1236-2"
comment = "final-project-part4-winningteamnamept4"

# generate passcode
callsignr = callsign.split('-')[0].upper()
code = 0x73e2
for i, char in enumerate(callsignr):
    code ^= ord(char) << (8 if not i % 2 else 0)

passcode = code & 0x7fff
print("Passcode:", passcode)

cmd = "cat direwolf-loopback-DRAFT.conf  | sed  's/EE123_CALL/"+callsign+"/g'  | "
cmd = cmd +  "sed  's/EE123_PASSCODE/%d/g' | " % (passcode)
cmd = cmd +  "sed  's/EE123_COMMENT/comment=\"%s\"/g' | " % (comment)
cmd = cmd +  "sed  's/EE123_LATCOORD/%s/g' | " % (lat)
cmd = cmd +  "sed  's/EE123_LONGCOORD/%s/g' > direwolf-loopback.conf" % (long)
print(cmd)
os.system(cmd)





