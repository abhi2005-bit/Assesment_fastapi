# 🚀 Online Assessment System

Enterprise-grade AI-powered assessments with multi-LLM fallback support.

## 📋 Features

- **Multi-LLM Support**: Gemini, Groq, OpenRouter, Perplexity with automatic fallback
- **Modern UI/UX**: Clean, elegant interface with responsive design
- **Real-time Assessment**: Timer-based assessments with instant results
- **Data Persistence**: JSON-based result storage and analysis
- **Error Handling**: Robust error handling with graceful degradation
- **API Management**: Secure API key management with environment variables

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI Models**: Google Gemini, Groq, OpenRouter, Perplexity
- **Data Storage**: JSON files
- **Environment**: Python 3.8+

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kamlesh9876/Assement-Test.git
   cd Assement-Test
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### API Keys Required
Add the following to your `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

### Free API Keys Setup
- **Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Groq**: Get from [Groq Console](https://console.groq.com/keys)
- **OpenRouter**: Get from [OpenRouter](https://openrouter.ai/keys)
- **Perplexity**: Get from [Perplexity](https://www.perplexity.ai/settings/api)

## 🎯 Usage

1. **Registration**: Fill in candidate details
2. **Rules**: Read and accept assessment guidelines
3. **Assessment**: Complete 20 questions in 20 minutes
4. **Results**: View detailed performance analysis

## 📊 Assessment Features

- **20 Questions**: Multiple choice format
- **20 Minutes**: Fixed duration assessment
- **No Negative Marking**: Safe for attempting all questions
- **Instant Results**: Real-time score calculation
- **Detailed Analytics**: Performance breakdown by topic

## 🔄 Multi-LLM Fallback System

The system automatically tries multiple AI providers in priority order:

1. **Groq** (Fast, reliable)
2. **OpenRouter** (Multiple models)
3. **Perplexity** (Advanced reasoning)
4. **Gemini** (Google's model)
5. **Default Questions** (Fallback if all APIs fail)

## 🎨 UI/UX Design

- **Modern Design**: Clean, professional interface
- **Responsive**: Works on all devices
- **Accessible**: High contrast, readable typography
- **Interactive**: Smooth animations and transitions
- **User-Friendly**: Intuitive navigation

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
└── data/                # Data storage directory
    ├── results/         # Assessment results
    └── knowledge_base/  # Reference materials
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
# Deploy to https://share.streamlit.io/

# Using Docker
docker build -t assessment-system .
docker run -p 8501:8501 assessment-system
```

## 🔒 Security Features

- **Environment Variables**: Secure API key storage
- **Session Isolation**: User data privacy
- **Input Validation**: Prevents malicious input
- **Error Handling**: No sensitive data exposure

## 📈 Performance

- **Fast Loading**: Optimized asset delivery
- **Efficient Caching**: Reduces API calls
- **Responsive Design**: Works on all devices
- **Error Recovery**: Graceful degradation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the FAQ section

## 🔄 Updates

- **v1.0.0**: Initial release with basic assessment functionality
- **v1.1.0**: Added multi-LLM support
- **v1.2.0**: Enhanced UI/UX design
- **v1.3.0**: Improved error handling and fallback system

---

**Built with ❤️ for modern assessment needs**
