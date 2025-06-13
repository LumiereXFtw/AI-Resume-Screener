// Import necessary modules
const express = require('express');
const cors = require('cors');
const path = require('path');
const fetch = require('node-fetch'); // For making HTTP requests to external APIs

// Create an Express application instance
const app = express();
// Define the port the server will listen on. Use process.env.PORT for deployment,
// or default to 3000 for local development.
const PORT = process.env.PORT || 3000;

// Middleware setup:

// Enable CORS for all routes. This is crucial for allowing your frontend
// (which might be served from a different origin/port) to make requests to this backend.
app.use(cors());
// Parse JSON bodies for incoming requests. This allows you to receive JSON data
// from the frontend (e.g., resume text, job description).
app.use(express.json());
// Serve static files from the 'public' directory.
// This means if you place your HTML, CSS, and client-side JS files in a 'public' folder,
// Express will serve them automatically.
app.use(express.static(path.join(__dirname, 'public')));

// API Routes:

/**
 * GET /
 * Serves the main HTML file for the frontend application.
 * When a user navigates to the root URL, this sends back the index.html.
 */
app.get('/', (req, res) => {
    // Send the index.html file located in the 'public' directory.
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

/**
 * POST /screen-resume
 * This is the API endpoint the frontend will call to submit resume and job description
 * for screening.
 *
 * Currently, this endpoint acts as a proxy to the Gemini API. In a full-stack
 * implementation, this is where you would integrate calls to your Python
 * resume parsing and NLP services.
 */
app.post('/screen-resume', async (req, res) => {
    // Extract resumeText and jobDescription from the request body
    const { resumeText, jobDescription } = req.body;

    // Basic validation
    if (!resumeText || !jobDescription) {
        // If either is missing, send a 400 Bad Request response
        return res.status(400).json({ error: 'Resume text and job description are required.' });
    }

    try {
        // Construct the prompt for the Gemini API, similar to the frontend logic
        const prompt = `
            Act as an AI/ML resume screener.
            Compare the following resume with the job description and provide a concise summary.
            Include:
            1. A match score (e.g., "High Match", "Moderate Match", "Low Match").
            2. Key skills from the resume that align with the job description.
            3. Key requirements from the job description that are met by the resume.
            4. Any noticeable gaps or areas for improvement in the resume based on the job description.

            ---
            Resume:
            ${resumeText}

            ---
            Job Description:
            ${jobDescription}
        `;

        // Prepare the payload for the Gemini API request
        const payload = {
            contents: [{ role: "user", parts: [{ text: prompt }] }]
        };

        // IMPORTANT: In a real-world scenario, you would securely load your API key
        // from environment variables (e.g., process.env.GEMINI_API_KEY).
        // For this example, we're leaving it empty as the Canvas environment
        // will automatically provide it during execution.
        const apiKey = "";
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

        // Make the request to the Gemini API
        const geminiResponse = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        // Check if the Gemini API response was successful
        if (!geminiResponse.ok) {
            const errorData = await geminiResponse.json();
            console.error('Gemini API error:', errorData);
            // Forward the error from Gemini API back to the frontend
            return res.status(geminiResponse.status).json({
                error: `Gemini API error: ${errorData.error.message || 'Unknown error'}`
            });
        }

        const geminiResult = await geminiResponse.json();

        // Extract the text content from the Gemini response
        if (geminiResult.candidates && geminiResult.candidates.length > 0 &&
            geminiResult.candidates[0].content && geminiResult.candidates[0].content.parts &&
            geminiResult.candidates[0].content.parts.length > 0) {
            const screeningSummary = geminiResult.candidates[0].content.parts[0].text;
            // Send the AI-generated summary back to the frontend
            res.json({ success: true, summary: screeningSummary });
        } else {
            // Handle cases where Gemini response structure is unexpected
            res.status(500).json({ error: 'Failed to get a valid response from the AI model.' });
        }

    } catch (error) {
        console.error('Error screening resume:', error);
        // Send a 500 Internal Server Error response for unexpected errors
        res.status(500).json({ error: 'An internal server error occurred during screening.' });
    }
});

// Start the server and listen for incoming requests
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Serving static files from: ${path.join(__dirname, 'public')}`);
});
