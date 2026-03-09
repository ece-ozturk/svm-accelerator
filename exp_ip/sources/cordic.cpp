/* cordic.cpp - Evaluates CORDIC algorithm given quantised and transformed inputs*/
#include "cordic.h"
#include "exp_ip.h"
// Initialise table for iteration sequence, accounting for repeats
int m_seq[] = {
		1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16
};

// Initialise table containing fixed theta values
data_t theta_table[] ={
		0.54930614433405489105,	0.25541281188299536087,
		0.12565721414045305515,	0.06258157147700300904,
		0.06258157147700300904,	0.03126017849066699272,
		0.01562627175205221278,	0.00781265895154042121,
		0.00390626986839682621,	0.00195312748353255019,
		0.00097656281044103594,	0.00048828128880511288,
		0.00024414062985063861,	0.00012207031310632982,
		0.00012207031310632982,	0.00006103515632579122,
		0.00003051757813447390,	0.00001525878906368424
};

/* -----CORDIC Function to calculate exp(z)-----
 * This function assumes the input z has been quantised to fixed point precision defined by data_t
 * This precision level is calculated by Monte-Carlo Simulation
 * Internal and Output values are given wider precision width to account for bit shifts */

output_t cordic(data_t z)
{
	// Declare relevant variables
	int i, d; 						// Internal variables
	int N = 18; 					// Iteration count (including repeats E.g. 1, 2, 2 counts as 3 iterations
	internal_t x = K_n;				// Assign x to 1/K value
	internal_t y = 0;				// Assign y to 0
	internal_t x_shift, y_shift;	// Variables to hold shifted values
	output_t output;				// Output variable

	// Begin CORDIC algorithm, looping through N iterations
	for (i=0; i<N; i++)
	{
		// Shift values according to iteration number, repeats handled
		x_shift = x >> m_seq[i];
		y_shift = y >> m_seq[i];

		// Determine d value based on if current z is < 0
		if (z < 0) {d = -1;} 		// Vector is negative, rotate up
		else {d = 1;}				// Vector is positive, rotate down

		// Update x, y, z
		x += d * y_shift;			// Given u = -1 for hyperbolic, add shifted value
		y += d * x_shift;			// Follows algorithm, add shifted value
		z -= d * theta_table[i];	// Follows algorithm
	}

	// Calculate Output
	output = x + y;
	return output;
}
