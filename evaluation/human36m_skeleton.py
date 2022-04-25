# Code adopted from https://github.com/tomasjakab/keypointgan

hip = 'hip'
thorax = 'thorax'
r_hip = 'r_hip'
r_knee = 'r_knee'
r_ankle = 'r_ankle'
r_ball = 'r_ball'
r_toes = 'r_toes'
l_hip = 'l_hip'
l_knee = 'l_knee'
l_ankle = 'l_ankle'
l_ball = 'l_ball'
l_toes = 'l_toes'
neck_base = 'neck'
head_center = 'head-center'
head_back = 'head-back'
l_uknown = 'l_uknown'
l_shoulder = 'l_shoulder'
l_elbow = 'l_elbow'
l_wrist = 'l_wrist'
l_wrist_2 = 'l_wrist_2'
l_thumb = 'l_thumb'
l_little = 'l_little'
l_little_2 = 'l_little_2'
r_uknown = 'r_uknown'
r_shoulder = 'r_shoulder'
r_elbow = 'r_elbow'
r_wrist = 'r_wrist'
r_wrist_2 = 'r_wrist_2'
r_thumb = 'r_thumb'
r_little = 'r_little'
r_little_2 = 'r_little_2'
pelvis = 'pelvis'

links = (
    (r_hip, thorax),
    # (r_hip, pelvis),
    (r_knee, r_hip),
    (r_ankle, r_knee),
    (r_ball, r_ankle),
    (r_toes, r_ball),
    (l_hip, thorax),
    # (l_hip, pelvis),
    (l_knee, l_hip),
    (l_ankle, l_knee),
    (l_ball, l_ankle),
    (l_toes, l_ball),
    (neck_base, thorax),
    # (head_center, head_back),
    # (head_back, neck_base),
    # (head_back, head_center),
    # (head_center, neck_base),
    (head_back, neck_base),
    (head_center, head_back),

    (l_shoulder, neck_base),
    (l_elbow, l_shoulder),
    (l_wrist, l_elbow),
    (l_thumb, l_wrist),
    (l_little, l_wrist),
    (r_shoulder, neck_base),
    (r_elbow, r_shoulder),
    (r_wrist, r_elbow),
    (r_thumb, r_wrist),
    (r_little, r_wrist),
    # (pelvis, thorax),
)

links_simple = (
    (r_hip, thorax),
    # (r_hip, pelvis),
    (r_knee, r_hip),
    (r_ankle, r_knee),
    (r_ball, r_ankle),
    (r_toes, r_ball),
    (l_hip, thorax),
    # (l_hip, pelvis),
    (l_knee, l_hip),
    (l_ankle, l_knee),
    (l_ball, l_ankle),
    (l_toes, l_ball),
    (neck_base, thorax),
    # (head_center, head_back),
    # (head_back, neck_base),
    # (head_back, head_center),
    # (head_center, neck_base),
    (head_back, neck_base),
    (head_center, head_back),

    (l_shoulder, neck_base),
    (l_elbow, l_shoulder),
    (l_wrist, l_elbow),
    (r_shoulder, neck_base),
    (r_elbow, r_shoulder),
    (r_wrist, r_elbow),
    # (pelvis, thorax),
)

links_simple2 = (
    (r_hip, pelvis),
    (r_knee, r_hip),
    (r_ankle, r_knee),
    (r_toes, r_ankle),

    (l_hip, pelvis),
    (l_knee, l_hip),
    (l_ankle, l_knee),
    (l_toes, l_ankle),

    (neck_base, pelvis),
    (head_back, neck_base),

    (l_shoulder, neck_base),
    (l_elbow, l_shoulder),
    (l_wrist, l_elbow),

    (r_shoulder, neck_base),
    (r_elbow, r_shoulder),
    (r_wrist, r_elbow),
)

joint_indices = {
    hip: 0,
    thorax: 12,
    r_hip: 1,
    r_knee: 2,
    r_ankle: 3,
    r_ball: 4,
    r_toes: 5,

    l_hip: 6,
    l_knee: 7,
    l_ankle: 8,
    l_ball: 9,
    l_toes: 10,

    neck_base: 13,
    head_center: 14,
    head_back: 15,

    l_uknown: 16,
    l_shoulder: 17,
    l_elbow: 18,
    l_wrist: 19,
    l_wrist_2: 20,
    l_thumb: 21,
    l_little: 22,
    l_little_2: 23,

    r_uknown: 24,
    r_shoulder: 25,
    r_elbow: 26,
    r_wrist: 27,
    r_wrist_2: 28,
    r_thumb: 29,
    r_little: 30,
    r_little_2: 31,
    pelvis: 11
}

joints_eval_martinez = {
    'Hip': 0,
    'RHip': 1,
    'RKnee': 2,
    'RFoot': 3,
    'LHip': 6,
    'LKnee': 7,
    'LFoot': 8,
    'Spine': 12,
    'Thorax': 13,
    'Neck/Nose': 14,
    'Head': 15,
    'LShoulder': 17,
    'LElbow': 18,
    'LWrist': 19,
    'RShoulder': 25,
    'RElbow': 26,
    'RWrist': 27
}


official_eval = {
    'Pelvis': (pelvis),
    'RHip': (r_hip),
    'RKnee': (r_knee),
    'RAnkle': (r_ankle),
    'LHip': (l_hip),
    'LKnee': (l_knee),
    'LAnkle': (l_ankle),
    'Spine1': (thorax),
    'Neck': (head_center),
    'Head': (head_back),
    'Site': (neck_base),
    'LShoulder': (l_shoulder),
    'LElbow': (l_elbow),
    'LWrist': (l_wrist),
    'RShoulder': (r_shoulder),
    'RElbow': (r_elbow),
    'RWrist': (r_wrist)}


official_eval_indices = {k: joint_indices[v] for k, v in official_eval.items()}



def get_link_indices(links):
  return [(joint_indices[x], joint_indices[y]) for x, y in links]

simple_link_indices = get_link_indices(links_simple)
simple2_link_indices = get_link_indices(links_simple2)
link_indices = get_link_indices(links)


def get_lr_correspondences():
    paired = []
    for limb in joint_indices.keys():
        if limb[:2] == 'l_':
            paired.append(limb[2:])
    correspond = []
    for limb in paired:
        correspond.append((joint_indices['l_' + limb], joint_indices['r_' + limb]))
    return correspond

def get_simple_lr_correspondences():
    
    correspond = [(0,3), (1,4), (2,5), (10,13), (11,14), (12,15)]  # manually assign correspondences
    
    return correspond