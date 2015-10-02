
function strip(str)
  return str:match( "^%s*(.-)%s*$" )
end  
  
  
str = " \t \r \n String with spaces  \t  \r  \n  "

print(strip(str))
