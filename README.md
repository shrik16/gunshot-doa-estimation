## gunshot-doa-estimation

# **Microphone Array-Based Direction of Arrival (DoA) Estimation for Gunshot Detection**


### ***Overview***

This project implements a Direction of Arrival (DoA) estimation system for gunshot detection using a microphone array. It uses the Generalized Cross-Correlation with Phase Transform (GCC-PHAT) method to estimate the direction of a gunshot sound source in a simulated environment with added noise and reverberation. Visualizations are included to help understand the results.

### ***Disclaimer***

This project is intended for educational purposes only. It simulates gunshot direction of arrival (DoA) estimation in a controlled environment and should not be relied upon for real-world applications, such as security or law enforcement, due to its limitations, including the use of synthetic data, simplified assumptions, and lack of real-world validation.


### ***Features***

Simulates a tetrahedral microphone array with 4 microphones.

Uses GCC-PHAT to estimate the DoA of a gunshot signal.

Includes noise and reverberation simulation to mimic real-world conditions.

Outputs the estimated azimuth and elevation angles.

Visualizes the microphone array and DoA estimates in 3D.

Plots the received signals at each microphone over time.


### ***Requirements***

Python 3.8+

Dependencies listed in requirements.txt

### ***Installation***

Clone the repository:

bash
```

git clone https://github.com/shrik16/gunshot-doa-estimation.git
cd gunshot-doa-estimation
```
Create a virtual environment and activate it:

```

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:


```

pip install -r requirements.txt
```

Usage
Run the main script to simulate DoA estimation and generate visualizations:

```

python main.py
```

The script will:

**Simulate a gunshot signal, add noise and reverberation, and estimate the DoA.**

**Print the results (azimuth and elevation angles) to the console.**

**Save two plots in the output folder:**

**output/microphone_array_doa.png: A 3D plot showing the microphone array, true direction, and estimated direction.**

**output/received_signals.png: A time-domain plot of the signals received at each microphone.**


### ***Project Structure***

main.py: Main script for DoA estimation and visualization.

requirements.txt: List of Python dependencies.

README.md: Project documentation.

output/: Folder where visualization plots are saved (created automatically).


### ***Methodology***

Microphone Array: A tetrahedral array with 4 microphones is simulated. Positions are defined in 3D space.

Signal Simulation: A synthetic gunshot signal (impulsive, wideband) is generated with a known direction.

Noise and Reverberation: Gaussian noise and simulated reflections (reverberation) are added to mimic real-world conditions.

GCC-PHAT: The Generalized Cross-Correlation with Phase Transform method is used to compute time differences of arrival (TDOA) between microphone pairs.

DoA Estimation: TDOA values are used to estimate the azimuth and elevation angles of the sound source.

Visualization: Uses matplotlib to plot the microphone array in 3D and the received signals over time.


### ***Results***

The script outputs the estimated DoA in terms of azimuth and elevation angles. Visualizations provide a clearer understanding of the spatial arrangement and signal characteristics.


### ***Limitations***

This is a simulation-based project and does not use real microphone data.

Performance may vary in real-world scenarios with complex noise and reverberation.

The tetrahedral array assumes a simplified environment.


### ***Future Improvements***

Add preprocessing to handle reverberation (e.g., dereverberation techniques).

Integrate real microphone data using hardware like Arduino or Raspberry Pi.

Extend to multiple sound sources.


**MIT License - feel free to use and modify this code.**



### ***Author***

**ShriK**
