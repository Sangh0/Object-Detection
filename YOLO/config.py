MISSING_IDX = []
NUM_CLASSES = 26

SCALES = 3
NUM_ANCHORS_PER_SCALE = 3
ANCHORS = [(10, 13), (16, 30), (33, 23), 
           (30, 61), (62, 45), (59, 119), 
           (116, 90), (156, 198), (373, 326)] # small objects, medium objects, large objects
           
assert len(ANCHORS) == SCALES * NUM_ANCHORS_PER_SCALE
NUM_ATTRIB = 4 + 1 + NUM_CLASSES # box coordinates 4, objectness 1
LAST_LAYER_DIM = NUM_ANCHORS_PER_SCALE * NUM_ATTRIB

IGNORE_THRESH = 0.5
NOOBJ_COEFF = 0.2
COORD_COEFF = 5

EPSILON = 1e-9