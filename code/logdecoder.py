
import struct
import sys

class BinaryReaderEOFException(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'Not enough bytes in file to satisfy read request'

# class BinaryReader:
#     # Map well-known type names into struct format characters.
#     typeNames = {
#         'int8'   :'b',
#         'uint8'  :'B',
#         'int16'  :'h',
#         'uint16' :'H',
#         'int32'  :'i',
#         'uint32' :'I',
#         'int64'  :'q',
#         'uint64' :'Q',
#         'float'  :'f',
#         'double' :'d',
#         'char'   :'s'}

#     def __init__(self, fileName, ending='>'):
#         self.file = open(fileName, 'rb')
#         self.ending = ending

#     def read(self, typeName):
#         typeFormat = BinaryReader.typeNames[typeName.lower()]
#         typeSize = struct.calcsize(typeFormat)
#         value = self.file.read(typeSize)
#         if typeSize != len(value):
#             raise BinaryReaderEOFException
#         return struct.unpack(typeFormat, value)[0]
    
#     def __del__(self):
#         self.file.close()

class LogParser:
    typeNames = {
        'int8'   :'b',
        'uint8'  :'B',
        'int16'  :'h',
        'uint16' :'H',
        'int32'  :'i',
        'uint32' :'I',
        'int64'  :'q',
        'uint64' :'Q',
        'float'  :'f',
        'double' :'d',
        'char'   :'s'
        }

    def __init__(self, fileName):
        self.fid = open(fileName, 'rb')
        self.logData = {}
    
    def read(self, sizeA, fmt,ending='>'):
        typeFormat = LogParser.typeNames[fmt.lower()]
        expected_size = struct.calcsize(typeFormat)
        value = self.fid.read(expected_size * sizeA)
        #print("value:{}", value)
        if expected_size * sizeA != len(value):
            raise BinaryReaderEOFException
            #return None

        if sizeA==1:
            typeFormat = "{}{}".format(ending,typeFormat)
        else:
            typeFormat = "{}{}".format(sizeA,typeFormat)

        #print("type:{}".format(typeFormat))
        return struct.unpack(typeFormat, value)[0]

    def parse(self):
        self.logData['fileVersion'] = self.read(1, 'uint16' )
        self.logData['taskID'] = self.read(1,'uint8')
        self.logData['taskVersion'] = self.read(1,'uint8')
        self.logData['subject'] = (self.read(10, 'char')).strip()
        self.logData['date'] = self.read(8, 'char')
        self.logData['startTime'] = self.read(5, 'char')

        self.fid.seek(2*1024, 0)
        self.logData['comment'] = self.read(1024,'char').strip()


        self.fid.seek(3*1024, 0)
        fullHeaderString = self.read(1024,'char').strip()
        end_point = fullHeaderString.index(b'\r\n')
        print("end_point:{}".format(end_point))
        fullHeaderString = fullHeaderString[0:end_point]
        headers = fullHeaderString.split(b",")
        #self.logData['headers']  = headers

        self.fid.seek(4*1024, 0)
        for i in range(len(headers)):
            self.logData[headers[i]] = self.read(1,'double')

        self.fid.seek(5*1024, 0)
        fullHeaderString = self.read(1024,'char').strip()
        end_point = fullHeaderString.index(b'\r\n')
        print("end_point:{}".format(end_point))
        fullHeaderString = fullHeaderString[0:end_point]
        headers = fullHeaderString.split(b",")
        #self.logData['headers']  = headers

        
        for header in headers:
            self.logData[header] = []
        
        self.fid.seek(6*1024, 0)
        while True:
            try:
                for header in headers:
                    data = self.read(1,'double')
                    self.logData[header].append(data)
            except:
                #print('End')
                break
            
        return self.logData

if __name__ == '__main__':
    if len(sys.argv)<2:
        print("python logdecoder.py <logfilename>")
        sys.exit(1)

    testfile=sys.argv[1]
    parser = LogParser(testfile)
    logData = parser.parse()
    print("logData:{}".format(logData))
    