const express = require('express');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
const dotenv = require('dotenv');
const FormData = require('form-data');
const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const mime = require('mime-types');
const { v4: uuidv4 } = require('uuid');

// Google Gemini SDK
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Load environment variables
dotenv.config();

// Initialize Express app
const app = express();
app.use(cors());
app.use(express.json());

// Create necessary directories
const createDirectories = async () => {
  const dirs = ['uploads', 'storage/resumes', 'data'];
  for (const dir of dirs) {
    try {
      await fs.mkdir(dir, { recursive: true });
    } catch (err) {
      console.log(`Directory ${dir} already exists or created`);
    }
  }
};

// Enhanced multer setup with file validation
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}-${file.originalname}`;
    cb(null, uniqueName);
  }
});

const fileFilter = (req, file, cb) => {
  const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only PDF, DOC, DOCX, and TXT files are allowed.'), false);
  }
};

const upload = multer({ 
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

const PYTHON_API = process.env.PYTHON_API || 'http://localhost:5001/api/screen';
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Enhanced Gemini Feedback Function
async function getGeminiFeedback(resumeText, jobDescription) {
  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

    const prompt = `
You are an expert AI resume screening assistant with deep knowledge of recruitment and HR practices.

Analyze the following resume against the job description and provide detailed feedback.

Resume Text:
"${resumeText}"

Job Description:
"${jobDescription}"

Please analyze and respond in the following JSON format:
{
  "summary": "Brief overall assessment of the candidate's fit for the role",
  "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
  "improvements": ["specific improvement area 1", "specific improvement area 2", "specific improvement area 3"],
  "missing_requirements": ["requirement 1", "requirement 2"],
  "match_score": 85,
  "experience_level": "junior/mid/senior",
  "recommended_next_steps": ["action 1", "action 2"],
  "salary_expectation": "estimated range based on skills and experience",
  "interview_questions": ["relevant question 1", "relevant question 2", "relevant question 3"]
}

Ensure the match_score is between 0-100 and reflects realistic assessment.
`;

    const result = await model.generateContent(prompt);
    const text = result.response.text();
    
    // Clean the response to extract JSON
    let cleanedText = text.trim();
    if (cleanedText.includes('```json')) {
      cleanedText = cleanedText.replace(/```json\n?/g, '').replace(/```/g, '');
    }
    
    return JSON.parse(cleanedText);
  } catch (err) {
    console.error('Gemini API error:', err.message);
    return {
      summary: "Unable to generate AI feedback at this time",
      strengths: [],
      improvements: [],
      missing_requirements: [],
      match_score: 0,
      experience_level: "unknown",
      recommended_next_steps: [],
      salary_expectation: "Unable to estimate",
      interview_questions: []
    };
  }
}

// Local data storage functions
const DATA_FILE = path.join(__dirname, 'data', 'screenings.json');

async function loadScreenings() {
  try {
    const data = await fs.readFile(DATA_FILE, 'utf8');
    return JSON.parse(data);
  } catch (err) {
    return [];
  }
}

async function saveScreening(screening) {
  try {
    const screenings = await loadScreenings();
    screenings.push(screening);
    await fs.writeFile(DATA_FILE, JSON.stringify(screenings, null, 2));
    return true;
  } catch (err) {
    console.error('Error saving screening:', err.message);
    return false;
  }
}

// Enhanced analytics function
function calculateAnalytics(screenings) {
  if (screenings.length === 0) return null;
  
  const scores = screenings.map(s => s.gemini_feedback?.match_score || 0);
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  
  const topCandidates = screenings
    .filter(s => (s.gemini_feedback?.match_score || 0) >= 70)
    .sort((a, b) => (b.gemini_feedback?.match_score || 0) - (a.gemini_feedback?.match_score || 0))
    .slice(0, 5);
  
  const commonSkills = {};
  screenings.forEach(s => {
    if (s.matched_skills) {
      s.matched_skills.forEach(skill => {
        commonSkills[skill] = (commonSkills[skill] || 0) + 1;
      });
    }
  });
  
  return {
    totalApplications: screenings.length,
    averageScore: Math.round(avgScore * 100) / 100,
    topCandidates: topCandidates.map(c => ({
      id: c.id,
      name: c.resume_name,
      score: c.gemini_feedback?.match_score || 0,
      timestamp: c.timestamp
    })),
    mostCommonSkills: Object.entries(commonSkills)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([skill, count]) => ({ skill, count }))
  };
}

// POST /api/screen - Enhanced screening endpoint
app.post('/api/screen', upload.single('resume'), async (req, res) => {
  try {
    if (!req.file || !req.body.job_description) {
      return res.status(400).json({ error: 'Missing resume file or job description' });
    }

    const jobDescription = req.body.job_description;
    const filePath = req.file.path;
    const originalName = req.file.originalname;
    const uniqueID = uuidv4();
    
    // Copy file to permanent storage
    const extension = path.extname(originalName);
    const permanentFileName = `${uniqueID}${extension}`;
    const permanentPath = path.join(__dirname, 'storage', 'resumes', permanentFileName);
    
    await fs.copyFile(filePath, permanentPath);

    // Send to Python microservice
    const form = new FormData();
    form.append('resume', fsSync.createReadStream(filePath), originalName);
    form.append('job_description', jobDescription);

    console.log('Sending request to Python API:', PYTHON_API);
    
    const response = await axios.post(PYTHON_API, form, {
      headers: form.getHeaders(),
      timeout: 30000 // 30 second timeout
    });

    const { tfidf_score, spacy_score, matched_skills, resume_text, gemini_analysis } = response.data;

    // Get enhanced Gemini AI Feedback
    const geminiFeedback = await getGeminiFeedback(resume_text || "", jobDescription);

    // Clean up temp file
    await fs.unlink(filePath);

    // Prepare screening data
    const screeningData = {
      id: uniqueID,
      timestamp: new Date().toISOString(),
      job_description: jobDescription,
      resume_name: originalName,
      resume_path: permanentPath,
      resume_url: `/api/resume/${uniqueID}`,
      tfidf_score: parseFloat(tfidf_score) || 0,
      spacy_score: parseFloat(spacy_score) || 0,
      matched_skills: matched_skills || [],
      resume_text: resume_text || "",
      python_gemini_analysis: gemini_analysis || "",
      gemini_feedback: geminiFeedback
    };

    // Save to local storage
    const saved = await saveScreening(screeningData);
    if (!saved) {
      console.warn('Failed to save screening data locally');
    }

    // Remove sensitive data from response
    const { resume_text: _, resume_path: __, ...responseData } = screeningData;

    res.json(responseData);

  } catch (err) {
    console.error('Server Error:', err.message);
    
    // Clean up temp file if it exists
    if (req.file?.path) {
      try {
        await fs.unlink(req.file.path);
      } catch (cleanupErr) {
        console.error('Error cleaning up temp file:', cleanupErr.message);
      }
    }

    if (err.code === 'ECONNREFUSED') {
      res.status(503).json({ error: 'Python analysis service is not available' });
    } else if (err.message.includes('timeout')) {
      res.status(504).json({ error: 'Analysis timeout - please try again' });
    } else {
      res.status(500).json({ error: 'Something went wrong on the server.' });
    }
  }
});

// GET /api/screenings - Get all screenings with pagination
app.get('/api/screenings', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const sortBy = req.query.sortBy || 'timestamp';
    const order = req.query.order || 'desc';
    
    const screenings = await loadScreenings();
    
    // Sort screenings
    screenings.sort((a, b) => {
      if (order === 'desc') {
        return new Date(b[sortBy]) - new Date(a[sortBy]);
      } else {
        return new Date(a[sortBy]) - new Date(b[sortBy]);
      }
    });
    
    // Paginate
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedScreenings = screenings.slice(startIndex, endIndex);
    
    // Remove sensitive data
    const cleanedScreenings = paginatedScreenings.map(({ resume_text, resume_path, ...rest }) => rest);
    
    res.json({
      data: cleanedScreenings,
      pagination: {
        current_page: page,
        total_pages: Math.ceil(screenings.length / limit),
        total_items: screenings.length,
        items_per_page: limit
      }
    });
  } catch (err) {
    console.error('Error fetching screenings:', err.message);
    res.status(500).json({ error: 'Failed to fetch screenings' });
  }
});

// GET /api/screening/:id - Get specific screening
app.get('/api/screening/:id', async (req, res) => {
  try {
    const screenings = await loadScreenings();
    const screening = screenings.find(s => s.id === req.params.id);
    
    if (!screening) {
      return res.status(404).json({ error: 'Screening not found' });
    }
    
    // Remove sensitive file path but keep other data
    const { resume_path, ...responseData } = screening;
    res.json(responseData);
  } catch (err) {
    console.error('Error fetching screening:', err.message);
    res.status(500).json({ error: 'Failed to fetch screening' });
  }
});

// GET /api/resume/:id - Serve resume file
app.get('/api/resume/:id', async (req, res) => {
  try {
    const screenings = await loadScreenings();
    const screening = screenings.find(s => s.id === req.params.id);
    
    if (!screening || !screening.resume_path) {
      return res.status(404).json({ error: 'Resume not found' });
    }
    
    const filePath = screening.resume_path;
    const fileExists = await fs.access(filePath).then(() => true).catch(() => false);
    
    if (!fileExists) {
      return res.status(404).json({ error: 'Resume file not found on disk' });
    }
    
    const mimeType = mime.lookup(filePath) || 'application/octet-stream';
    res.setHeader('Content-Type', mimeType);
    res.setHeader('Content-Disposition', `inline; filename="${screening.resume_name}"`);
    
    const fileStream = fsSync.createReadStream(filePath);
    fileStream.pipe(res);
  } catch (err) {
    console.error('Error serving resume:', err.message);
    res.status(500).json({ error: 'Failed to serve resume' });
  }
});

// GET /api/analytics - Get screening analytics
app.get('/api/analytics', async (req, res) => {
  try {
    const screenings = await loadScreenings();
    const analytics = calculateAnalytics(screenings);
    
    if (!analytics) {
      return res.json({
        totalApplications: 0,
        averageScore: 0,
        topCandidates: [],
        mostCommonSkills: []
      });
    }
    
    res.json(analytics);
  } catch (err) {
    console.error('Error calculating analytics:', err.message);
    res.status(500).json({ error: 'Failed to calculate analytics' });
  }
});

// DELETE /api/screening/:id - Delete a screening
app.delete('/api/screening/:id', async (req, res) => {
  try {
    const screenings = await loadScreenings();
    const screeningIndex = screenings.findIndex(s => s.id === req.params.id);
    
    if (screeningIndex === -1) {
      return res.status(404).json({ error: 'Screening not found' });
    }
    
    const screening = screenings[screeningIndex];
    
    // Delete the resume file
    if (screening.resume_path) {
      try {
        await fs.unlink(screening.resume_path);
      } catch (fileErr) {
        console.warn('Could not delete resume file:', fileErr.message);
      }
    }
    
    // Remove from array and save
    screenings.splice(screeningIndex, 1);
    await fs.writeFile(DATA_FILE, JSON.stringify(screenings, null, 2));
    
    res.json({ message: 'Screening deleted successfully' });
  } catch (err) {
    console.error('Error deleting screening:', err.message);
    res.status(500).json({ error: 'Failed to delete screening' });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    python_api: PYTHON_API
  });
});

// Initialize directories and start server
const PORT = process.env.PORT || 5000;

createDirectories().then(() => {
  app.listen(PORT, () => {
    console.log(`Node server listening at http://localhost:${PORT}`);
    console.log(`Python API configured at: ${PYTHON_API}`);
    console.log('Available endpoints:');
    console.log('  POST /api/screen - Screen a resume');
    console.log('  GET /api/screenings - Get all screenings');
    console.log('  GET /api/screening/:id - Get specific screening');
    console.log('  GET /api/resume/:id - View resume file');
    console.log('  GET /api/analytics - Get analytics');
    console.log('  DELETE /api/screening/:id - Delete screening');
    console.log('  GET /api/health - Health check');
  });
}).catch(err => {
  console.error('Failed to initialize directories:', err);
  process.exit(1);
});