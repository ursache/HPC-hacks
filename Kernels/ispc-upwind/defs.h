#define INPUTSTRIDE (32 + 4)
#define INPUTSLICE  (INPUTSTRIDE * INPUTSTRIDE)
#define INPUTVOLUME (INPUTSLICE * INPUTSTRIDE)

#define OUTPUTSTRIDE 32
#define OUTPUTSLICE (OUTPUTSTRIDE * OUTPUTSTRIDE)
#define OUTPUTVOLUME (OUTPUTSLICE * OUTPUTSTRIDE)

typedef REAL real;
