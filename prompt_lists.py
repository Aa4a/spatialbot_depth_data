

depth_value_prompts = [
    '''
    Describe a Depth Map.
    ''',

    '''
    Compare the far and near view based on depth values.
    '''
]


spatial_understanding_prompts = [
    '''
    From the photographer's(oraudience's) perspective, the positional relationship of objects(left/right, down/up, front/back).
    ''',

    '''
    Distance (which is farther/closer to something).
    ''',

    '''
    Size (big/small, tall/short, wide/narrow).
    ''',

    '''
    Contact (does A touch/contact B).
    ''',

    '''
    Arrangement of objects.
    ''',

    '''
    Top/bottom, inside/outside relationships, etc.
    '''
]


scene_understanding_prompts = [
    '''
    What is the robot doing and how should it complete the task?
    ''',

    '''
    Object state:
    Orientation, stable placement, and inversion.
    ''',

    '''
    Number/color of objects.
    ''',

    '''
    object position:
    From the photographer's(or audience's) perspective, where is it in the scene.
    ''',

    '''
    Object appearance.
    ''',

    '''
    Obstacle detection:
    Is there an obstacle ahead? Is the object obstructed and cannot be reached?
    '''
]