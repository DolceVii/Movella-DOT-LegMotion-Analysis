#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Εισαγωγή βιβλιοθηκών.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Φόρτωση δεδομένων από το Excel.
file_path = 'all_data_excel_files.xlsx'
thigh_data = pd.read_excel(file_path, sheet_name='Thigh')
shin_data = pd.read_excel(file_path, sheet_name='Shin')
toe_data = pd.read_excel(file_path, sheet_name='Toe')

# Υποθέτουμε ότι τα δεδομένα έχουν στήλες: timestamp, euler.x, euler.y, euler.z
time = thigh_data['Timestamp']
thigh_euler = thigh_data[['Euler.x', 'Euler.y', 'Euler.z']]
shin_euler = shin_data[['Euler.x', 'Euler.y', 'Euler.z']]
toe_euler = toe_data[['Euler.x', 'Euler.y', 'Euler.z']]

# Έλεγχος για την εξασφάλιση ισομήκων δεδομένων.
min_length = min(len(thigh_euler), len(shin_euler), len(toe_euler))
thigh_euler = thigh_euler.iloc[:min_length]
shin_euler = shin_euler.iloc[:min_length]
toe_euler = toe_euler.iloc[:min_length]
time = time.iloc[:min_length]

# Σταθερά μήκη τμημάτων ποδιού.
L_thigh = 0.45  # Μήκος μπούτι.
L_shin = 0.40   # Μήκος κνήμη.
L_foot = 0.20   # Μήκος πέλμα.

# Συντελεστές για την αύξηση της κίνησης.
thigh_movement_factor = 1.5
shin_movement_factor = 1.5
toe_movement_factor = 2

# Συνάρτηση για τον υπολογισμό των θέσεων των αρθρώσεων σε 3D.
def calculate_joint_positions(thigh, shin, toe):
    hip_positions = []
    knee_positions = []
    ankle_positions = []
    toe_positions = []
    
    for i in range(len(thigh)):
        hip = np.array([0, 0, 0])
        
        thigh_angle = np.radians(thigh.iloc[i] * thigh_movement_factor)
        shin_angle = np.radians(shin.iloc[i] * shin_movement_factor)
        toe_angle = np.radians(toe.iloc[i] * toe_movement_factor)
        
        knee = hip + L_thigh * np.array([np.cos(thigh_angle.iloc[1]), np.sin(thigh_angle.iloc[1]), np.sin(thigh_angle.iloc[0])])
        ankle = knee + L_shin * np.array([np.cos(shin_angle.iloc[1]), np.sin(shin_angle.iloc[1]), np.sin(shin_angle.iloc[0])])
        toe_pos = ankle + L_foot * np.array([np.cos(toe_angle.iloc[1]), np.sin(toe_angle.iloc[1]), np.sin(toe_angle.iloc[0])])
        
        # Περιορισμός κίνησης του "toe" να μην πηγαίνει πολύ κάτω και προσθήκη offset για κεντράρισμα.
        if toe_pos[2] < ankle[2]:
            toe_pos[2] = ankle[2]
        toe_pos += np.array([0, 0, 0.2])  # Προσθήκη μεγαλύτερου offset.
        
        knee += np.array([0, 0, 0.1])
        ankle += np.array([0, 0, 0.1])
        
        hip_positions.append(hip)
        knee_positions.append(knee)
        ankle_positions.append(ankle)
        toe_positions.append(toe_pos)
    
    return np.array(hip_positions), np.array(knee_positions), np.array(ankle_positions), np.array(toe_positions)

# Υπολογισμός των θέσεων των αρθρώσεων.
hip_positions, knee_positions, ankle_positions, toe_positions = calculate_joint_positions(thigh_euler, shin_euler, toe_euler)

# Προετοιμασία δεδομένων για το μοντέλο ΑΙ.
X = np.hstack((thigh_euler, shin_euler, toe_euler))
y = np.zeros(len(X))  # Ετικέτες για ανίχνευση σφαλμάτων (0: σωστό, 1: λάθος).

# Εισαγωγή τυχαίων σφαλμάτων για εκπαιδευτικούς σκοπούς.
error_indices = np.random.choice(len(X), size=int(len(X) * 0.1), replace=False)
y[error_indices] = 1

# Διάσπαση δεδομένων σε εκπαιδευτικό και δοκιμαστικό σύνολο.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Βρόχος εκπαίδευσης και αξιολόγησης μοντέλου μέχρι να επιτευχθεί η επιθυμητή ακρίβεια.
desired_accuracy = 0.88
max_iterations = 100
iteration = 0
accuracy = 0

while accuracy < desired_accuracy and iteration < max_iterations:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    iteration += 1
    print(f'Iteration {iteration}, Accuracy: {accuracy:.2f}')

if accuracy >= desired_accuracy:
    print(f'Final model accuracy: {accuracy:.2f}')
else:
    print(f'Could not reach desired accuracy after {max_iterations} iterations. Final accuracy: {accuracy:.2f}')

# Συνάρτηση για τον έλεγχο της σωστότητας της άσκησης με χρήση ΑΙ.
def check_exercise_with_ai(X, model):
    predictions = model.predict(X)
    valid = np.all(predictions == 0)
    errors = np.where(predictions == 1)[0]
    return valid, errors

# Υπολογισμός προτεινόμενων διορθώσεων για κάθε άρθρωση.
def calculate_correction(thigh, shin, toe, errors, thigh_factor, shin_factor, toe_factor):
    corrected_positions = []
    for i in errors:
        corrected_thigh = thigh.iloc[i] / thigh_factor
        corrected_shin = shin.iloc[i] / shin_factor
        corrected_toe = toe.iloc[i] / toe_factor
        corrected_positions.append([corrected_thigh, corrected_shin, corrected_toe])
    return np.array(corrected_positions)

# Συντελεστές διόρθωσης.
thigh_correction_factor = thigh_movement_factor
shin_correction_factor = shin_movement_factor
toe_correction_factor = toe_movement_factor

# Έλεγχος της άσκησης.
valid, errors = check_exercise_with_ai(X, clf)
corrections = calculate_correction(thigh_euler, shin_euler, toe_euler, errors, thigh_correction_factor, shin_correction_factor, toe_correction_factor)

# Δημιουργία της προσομοίωσης.
fig = plt.figure(figsize=(14, 7))

# Προσθήκη 3D γραφήματος για τις θέσεις των αρθρώσεων.
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=20., azim=-60)

line_hip_knee, = ax.plot([], [], [], 'r-', lw=2, label='Hip to Knee')
line_knee_ankle, = ax.plot([], [], [], 'g-', lw=2, label='Knee to Ankle')
line_ankle_toe, = ax.plot([], [], [], 'b-', lw=2, label='Ankle to Toe')
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
validation_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, color='green' if valid else 'red')

# Προσθήκη ετικετών αξόνων και τίτλου.
ax.set_title('Leg Joint Movement Simulation')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Προσθήκη σφαιριδίων στις αρθρώσεις.
hip_marker, = ax.plot([], [], [], 'ro')
knee_marker, = ax.plot([], [], [], 'go')
ankle_marker, = ax.plot([], [], [], 'bo')
toe_marker, = ax.plot([], [], [], 'mo')

# Προσθήκη διακεκομμένων γραμμών διόρθωσης.
line_hip_knee_corrected, = ax.plot([], [], [], 'm--', lw=2, label='Corrected Hip to Knee')
line_knee_ankle_corrected, = ax.plot([], [], [], 'm--', lw=2, label='Corrected Knee to Ankle')
line_ankle_toe_corrected, = ax.plot([], [], [], 'm--', lw=2, label='Corrected Ankle to Toe')

# Προσθήκη διαγραμμάτων Euler για κάθε άρθρωση.
ax_euler = fig.add_subplot(322)
ax_euler.plot(time, thigh_euler['Euler.x'], 'r', label='Thigh Euler.x')
ax_euler.plot(time, thigh_euler['Euler.y'], 'g', label='Thigh Euler.y')
ax_euler.plot(time, thigh_euler['Euler.z'], 'b', label='Thigh Euler.z')
ax_euler.set_title('Thigh Euler Angles')
ax_euler.set_xlabel('Time')
ax_euler.set_ylabel('Angle (degrees)')
ax_euler.legend()

ax_euler_shin = fig.add_subplot(324)
ax_euler_shin.plot(time, shin_euler['Euler.x'], 'r', label='Shin Euler.x')
ax_euler_shin.plot(time, shin_euler['Euler.y'], 'g', label='Shin Euler.y')
ax_euler_shin.plot(time, shin_euler['Euler.z'], 'b', label='Shin Euler.z')
ax_euler_shin.set_title('Shin Euler Angles')
ax_euler_shin.set_xlabel('Time')
ax_euler_shin.set_ylabel('Angle (degrees)')
ax_euler_shin.legend()

ax_euler_toe = fig.add_subplot(326)
ax_euler_toe.plot(time, toe_euler['Euler.x'], 'r', label='Toe Euler.x')
ax_euler_toe.plot(time, toe_euler['Euler.y'], 'g', label='Toe Euler.y')
ax_euler_toe.plot(time, toe_euler['Euler.z'], 'b', label='Toe Euler.z')
ax_euler_toe.set_title('Toe Euler Angles')
ax_euler_toe.set_xlabel('Time')
ax_euler_toe.set_ylabel('Angle (degrees)')
ax_euler_toe.legend()

def init():
    # Αρχικοποίηση  γραμμών και κειμένου του χρόνου.
    line_hip_knee.set_data([], [])
    line_hip_knee.set_3d_properties([])
    line_knee_ankle.set_data([], [])
    line_knee_ankle.set_3d_properties([])
    line_ankle_toe.set_data([], [])
    line_ankle_toe.set_3d_properties([])
    line_hip_knee_corrected.set_data([], [])
    line_hip_knee_corrected.set_3d_properties([])
    line_knee_ankle_corrected.set_data([], [])
    line_knee_ankle_corrected.set_3d_properties([])
    line_ankle_toe_corrected.set_data([], [])
    line_ankle_toe_corrected.set_3d_properties([])
    time_text.set_text('')
    validation_text.set_text('')
    hip_marker.set_data([], [])
    hip_marker.set_3d_properties([])
    knee_marker.set_data([], [])
    knee_marker.set_3d_properties([])
    ankle_marker.set_data([], [])
    ankle_marker.set_3d_properties([])
    toe_marker.set_data([], [])
    toe_marker.set_3d_properties([])
    return (line_hip_knee, line_knee_ankle, line_ankle_toe, line_hip_knee_corrected, 
            line_knee_ankle_corrected, line_ankle_toe_corrected, time_text, 
            validation_text, hip_marker, knee_marker, ankle_marker, toe_marker)

def animate(i):
    # Ενημέρωση γραμμών.
    line_hip_knee.set_data([hip_positions[i, 0], knee_positions[i, 0]], [hip_positions[i, 1], knee_positions[i, 1]])
    line_hip_knee.set_3d_properties([hip_positions[i, 2], knee_positions[i, 2]])
    
    line_knee_ankle.set_data([knee_positions[i, 0], ankle_positions[i, 0]], [knee_positions[i, 1], ankle_positions[i, 1]])
    line_knee_ankle.set_3d_properties([knee_positions[i, 2], ankle_positions[i, 2]])
    
    line_ankle_toe.set_data([ankle_positions[i, 0], toe_positions[i, 0]], [ankle_positions[i, 1], toe_positions[i, 1]])
    line_ankle_toe.set_3d_properties([ankle_positions[i, 2], toe_positions[i, 2]])
    
    # Ενημέρωση σφαιριδίων.
    hip_marker.set_data([hip_positions[i, 0]], [hip_positions[i, 1]])
    hip_marker.set_3d_properties([hip_positions[i, 2]])
    
    knee_marker.set_data([knee_positions[i, 0]], [knee_positions[i, 1]])
    knee_marker.set_3d_properties([knee_positions[i, 2]])
    
    ankle_marker.set_data([ankle_positions[i, 0]], [ankle_positions[i, 1]])
    ankle_marker.set_3d_properties([ankle_positions[i, 2]])
    
    toe_marker.set_data([toe_positions[i, 0]], [toe_positions[i, 1]])
    toe_marker.set_3d_properties([toe_positions[i, 2]])
    
    # Ενημέρωση κειμένου χρόνου.
    time_text.set_text(f'Time = {time.iloc[i]:.2f}s')

    # Ενημέρωση κειμένου εγκυρότητας άσκησης.
    if valid:
        validation_text.set_text('The exercise is performed correctly.')
    else:
        validation_text.set_text('Errors found in the exercise.')

    # Επισήμανση καρέ με σφάλματα και αναφορά λεπτομερειών.
    if i in errors:
        validation_text.set_color('red')
        validation_text.set_text(f'Error at frame {i}: Joint angles out of bounds.')

        # Προσθήκη διακεκομμένων γραμμών διόρθωσης.
        corrected_positions = corrections[errors.tolist().index(i)]
        
        corrected_thigh_angle = np.radians(corrected_positions[0])
        corrected_shin_angle = np.radians(corrected_positions[1])
        corrected_toe_angle = np.radians(corrected_positions[2])
        
        corrected_knee = hip_positions[i] + L_thigh * np.array([np.cos(corrected_thigh_angle[1]), np.sin(corrected_thigh_angle[1]), np.sin(corrected_thigh_angle[0])])
        corrected_ankle = corrected_knee + L_shin * np.array([np.cos(corrected_shin_angle[1]), np.sin(corrected_shin_angle[1]), np.sin(corrected_shin_angle[0])])
        corrected_toe = corrected_ankle + L_foot * np.array([np.cos(corrected_toe_angle[1]), np.sin(corrected_toe_angle[1]), np.sin(corrected_toe_angle[0])])

        # Διατήρηση τιμών ώστε οι διακεκομμένες γραμμές να μην είναι χαμηλότερες από τις αρχικές.
        if corrected_knee[2] < knee_positions[i, 2]:
            corrected_knee[2] = knee_positions[i, 2]
        if corrected_ankle[2] < ankle_positions[i, 2]:
            corrected_ankle[2] = ankle_positions[i, 2]
        if corrected_toe[2] < toe_positions[i, 2]:
            corrected_toe[2] = toe_positions[i, 2]

        line_hip_knee_corrected.set_data([hip_positions[i, 0], corrected_knee[0]], [hip_positions[i, 1], corrected_knee[1]])
        line_hip_knee_corrected.set_3d_properties([hip_positions[i, 2], corrected_knee[2]])

        line_knee_ankle_corrected.set_data([corrected_knee[0], corrected_ankle[0]], [corrected_knee[1], corrected_ankle[1]])
        line_knee_ankle_corrected.set_3d_properties([corrected_knee[2], corrected_ankle[2]])

        line_ankle_toe_corrected.set_data([corrected_ankle[0], corrected_toe[0]], [corrected_ankle[1], corrected_toe[1]])
        line_ankle_toe_corrected.set_3d_properties([corrected_ankle[2], corrected_toe[2]])
    else:
        validation_text.set_color('green')
        
        # Απόκρυψη διακεκομμένων γραμμών όταν δεν υπάρχει λάθος.
        line_hip_knee_corrected.set_data([], [])
        line_hip_knee_corrected.set_3d_properties([])
        line_knee_ankle_corrected.set_data([], [])
        line_knee_ankle_corrected.set_3d_properties([])
        line_ankle_toe_corrected.set_data([], [])
        line_ankle_toe_corrected.set_3d_properties([])
    
    return (line_hip_knee, line_knee_ankle, line_ankle_toe, line_hip_knee_corrected, 
            line_knee_ankle_corrected, line_ankle_toe_corrected, time_text, 
            validation_text, hip_marker, knee_marker, ankle_marker, toe_marker)
# Ξεκινά το animation που δείχνει την κίνηση των αρθρώσεων με την πάροδο του χρόνου.
ani = FuncAnimation(fig, animate, init_func=init, frames=len(time), interval=50, blit=True)

ax.legend()
plt.tight_layout()
plt.show()

