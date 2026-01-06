# 🚀 Nelumbus AI Assessment System

Production-ready AI-powered assessment platform with real-time timer, adaptive questions, and comprehensive performance analytics.

## 🌟 Key Features

### 🎯 **Core Assessment System**
- **� 20 Questions** - Multiple choice format with adaptive generation
- **⏱️ Real-time Timer** - JavaScript countdown with color-coded warnings
- **� Performance Analytics** - Comprehensive results and recommendations
- **🤖 AI-Powered Questions** - Dynamic generation with multiple LLM providers
- **📷 Basic Camera** - Simple visual monitoring (320x240px)

### 🔧 **Multi-LLM Support**
- **🤖 Google Gemini** - Primary AI provider with fallback options
- **🦾 Groq** - Fast responses with llama models
- **🌐 OpenRouter** - Multiple model access
- **🔍 Perplexity** - Advanced reasoning capabilities
- **🔄 Automatic Fallback** - Seamless API switching on quota issues

### � **User Experience**
- **👤 Professional Registration** - Clean user onboarding
- **📋 Enhanced Rules** - Clear assessment guidelines
- **🎨 Modern UI** - Professional design with smooth transitions
- **📱 Responsive** - Works on all devices
- **⚡ Fast Performance** - Optimized for speed

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.28.1
- **Backend**: Python 3.8+
- **AI Models**: Gemini, Groq, OpenRouter, Perplexity
- **Data Storage**: JSON files with structured results
- **Camera**: Streamlit st.camera_input with CSS positioning
- **Timer**: JavaScript with real-time updates
- **Environment**: Cross-platform compatible

## 📦 Installation

### 1. **Clone the Repository**
```bash
git clone https://github.com/kamlesh9876/Assement-Test.git
cd Assement-Test
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. **Run the Application**
```bash
streamlit run app.py
```

## 🔧 Configuration

### API Keys Required
Add the following to your `.env` file:

```env
# Primary Gemini API Key
GEMINI_API_KEY=your_gemini_api_key

# Backup Gemini Keys (optional but recommended)
GEMINI_API_KEY_1=your_second_gemini_key
GEMINI_API_KEY_2=your_third_gemini_key

# Other AI Providers
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

### Assessment Flow
1. **👤 Registration** - Fill in candidate details (name, university, programming language, difficulty)
2. **📋 Rules** - Review and accept assessment guidelines
3. **🧠 Assessment** - Complete 20 questions with real-time timer
   - Live countdown timer with color warnings
   - Basic camera monitoring
   - Performance tracking
4. **📊 Results** - View comprehensive analysis with AI insights

### Key Features During Assessment
- **⏱️ Real-time Timer** - Counts down with visual warnings (green → orange → red)
- **📊 Live Progress** - Question tracking and completion status
- **📷 Camera Monitoring** - Basic visual monitoring in corner
- **🎯 Performance Metrics** - Real-time score calculation

## 🔄 API Management System

### Automatic Quota Handling
The system automatically manages API quotas:

1. **Priority Order**: Groq → OpenRouter → Perplexity → Gemini
2. **Auto-Rotation**: Switches APIs when quota exceeded
3. **Key Rotation**: Multiple Gemini keys for redundancy
4. **Graceful Fallback**: Uses default questions if all APIs fail

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example             # Environment variables template
├── README.md                # This documentation
├── TODO.txt                 # Future integration roadmap
├── data/                    # Data storage
│   ├── results/             # Assessment results
│   └── knowledge_base/     # Reference materials
├── integrated_assessment.py # Active assessment system
├── adaptive_agent.py        # Active AI question generation
├── advanced_proctoring.py   # Available for future integration
├── analytics_dashboard.py   # Available for future integration
├── computer_vision.py       # Available for future integration
├── conversational_ai.py     # Available for future integration
└── emotional_intelligence.py # Available for future integration
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run app.py --server.headless=true

# Using Docker
docker build -t nelumbus-assessment .
docker run -p 8501:8501 nelumbus-assessment
```

## 🔒 Security Features

- **Environment Variables**: Secure API key storage
- **Session Isolation**: User data privacy protection
- **Input Validation**: Prevents malicious input
- **Camera Control**: User consent and privacy controls
- **Error Handling**: No sensitive data exposure

## 📈 Performance Features

- **⚡ Fast Loading**: Optimized asset delivery
- **💾 Smart Caching**: Reduces API calls with hint caching
- **🔄 API Rotation**: Automatic load balancing
- **📱 Responsive Design**: Works on all devices
- **🛡️ Error Recovery**: Graceful degradation

##  Troubleshooting

### Common Issues

#### "API quota exceeded" Message
**Solution**: 
1. Wait for quota reset (usually 1 hour for free tiers)
2. Add backup API keys to `.env` file
3. Use multiple API providers for redundancy

#### Timer Not Working
**Solution**:
1. Ensure JavaScript is enabled in browser
2. Check browser console for errors
3. Refresh page and restart assessment

#### Questions Not Generating
**Solution**:
1. Check if API keys are correctly configured
2. Verify internet connection
3. System will fallback to default questions if APIs fail

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Support

For issues and questions:
- 🐛 Create an issue on GitHub
- 📧 Check the troubleshooting section
- � Review TODO.txt for planned features

## 🔄 Version History

- **v2.1.0**: Optimized Production Release
  - Simplified agent system (7 → 2 active agents)
  - Real-time JavaScript timer implementation
  - Enhanced assessment rules with professional UI
  - Comprehensive TODO.txt for future development
  - Code optimization (40% reduction in complexity)

- **v2.0.0**: Complete agentic assessment system
  - Added AI-powered user context tracking
  - Integrated corner camera monitoring
  - Implemented smart API quota management
  - Enhanced performance analytics

- **v1.3.0**: Improved error handling and fallback system
- **v1.2.0**: Enhanced UI/UX design
- **v1.1.0**: Added multi-LLM support
- **v1.0.0**: Initial release with basic assessment functionality

## 🎯 Current Status

**✅ Production Ready**: Core functionality fully operational
**🚀 Deployable**: Ready for production deployment
**📊 Optimized**: Simplified and maintainable codebase
**🔮 Future Roadmap**: Documented in TODO.txt

---

**Built with ❤️ for modern AI-powered assessment needs**

🌟 **Status**: Production Ready & Optimized
🔗 **Repository**: https://github.com/kamlesh9876/Assement-Test
� **Access**: http://localhost:8501
