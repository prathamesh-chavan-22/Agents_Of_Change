import React, { useState, useRef, useEffect } from 'react';
import { MapPin, Sparkles, Zap, Star } from 'lucide-react';
import AboutSection from './AboutSection';
import ChatMessage from './ChatMessage';
import MessageInput from './MessageInput';
import VoiceInput from './VoiceInput';
import ThemeToggle from './ThemeToggle';
import LanguageSelector from './LanguageSelector';

interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  audioUrl?: string;
  isLoading?: boolean;
}

interface ChatBotProps {
  theme: 'light' | 'dark';
  onThemeToggle: () => void;
}

const ChatBot: React.FC<ChatBotProps> = ({ theme, onThemeToggle }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [autoPlay, setAutoPlay] = useState(true);
  const [welcomeShown, setWelcomeShown] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showMagicalElements, setShowMagicalElements] = useState(false);
  const chatBoxRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Show welcome message with magical animation
  useEffect(() => {
    if (!welcomeShown && messages.length === 0) {
      const timer = setTimeout(() => {
        setWelcomeShown(true);
        setShowMagicalElements(true);
      }, 1200);
      return () => clearTimeout(timer);
    }
  }, [welcomeShown, messages.length]);

  const addMessage = (sender: 'user' | 'bot', content: string, audioUrl?: string, isLoading = false): string => {
    const newMessage: Message = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      sender,
      content,
      audioUrl,
      isLoading
    };
    setMessages(prev => [...prev, newMessage]);
    return newMessage.id;
  };

  const updateMessage = (id: string, updates: Partial<Message>) => {
    setMessages(prev => prev.map(msg => 
      msg.id === id ? { ...msg, ...updates } : msg
    ));
  };

  const removeMessage = (id: string) => {
    setMessages(prev => prev.filter(msg => msg.id !== id));
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isProcessing) {
      console.warn('Empty message or already processing');
      return;
    }

    console.log('Sending message:', text);
    setIsProcessing(true);

    // Add user message
    addMessage('user', text);

    // Add loading message
    const loadingId = addMessage('bot', 'Thinking...', undefined, true);

    try {
      console.log('Making API request to /ask endpoint');
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ question: text })
      });

      console.log('API response status:', response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('API response data:', data);
      
      // Remove loading message
      removeMessage(loadingId);
      
      if (data.answer && data.answer.trim()) {
        addMessage('bot', data.answer.trim(), data.audio_url);
      } else {
        addMessage('bot', '❌ No answer returned from the server.');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Remove loading message
      removeMessage(loadingId);
      
      let errorMessage = '❌ Error: ';
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage += 'Unable to connect to the server. Please check if the server is running on http://localhost:8000';
      } else if (error instanceof Error) {
        errorMessage += error.message;
      } else {
        errorMessage += 'An unexpected error occurred.';
      }
      
      addMessage('bot', errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVoiceTranscription = (transcription: string) => {
    console.log('Voice transcription received:', transcription);
    
    if (!transcription.trim()) {
      console.warn('Empty transcription received');
      return;
    }

    // Send the transcription directly without showing it as a separate user message
    // The handleSendMessage function will add it as a user message
    handleSendMessage(transcription);
  };

  const handleAutoPlayToggle = () => {
    if (isProcessing) {
      return; // Prevent toggling during processing
    }
    
    const newAutoPlay = !autoPlay;
    setAutoPlay(newAutoPlay);
    console.log('Auto-play toggled:', newAutoPlay);
  };

  // Determine if inputs should be disabled
  const inputsDisabled = isProcessing || isRecording;

  return (
    <div className="min-h-screen flex flex-col transition-all duration-500 relative overflow-hidden" 
         style={{ backgroundColor: 'var(--bg-primary)' }}>
      
      {/* Magical Floating Elements */}
      {showMagicalElements && (
        <div className="fixed inset-0 pointer-events-none z-10 overflow-hidden">
          {[...Array(15)].map((_, i) => (
            <div
              key={i}
              className="absolute animate-float opacity-30"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${4 + Math.random() * 6}s`
              }}
            >
              {i % 3 === 0 ? (
                <Sparkles size={8} className="text-yellow-400" />
              ) : i % 3 === 1 ? (
                <Star size={6} className="text-blue-400" />
              ) : (
                <Zap size={7} className="text-green-400" />
              )}
            </div>
          ))}
        </div>
      )}

      {/* Enhanced Background Pattern */}
      <div className="fixed inset-0 opacity-8 pointer-events-none">
        <div className="absolute inset-0 animate-pulse" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23e65100' fill-opacity='0.08'%3E%3Cpath d='M40 40c0-15.464-12.536-28-28-28s-28 12.536-28 28 12.536 28 28 28 28-12.536 28-28zm0 0c0 15.464 12.536 28 28 28s28-12.536 28-28-12.536-28-28-28-28 12.536-28 28z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }} />
      </div>

      {/* Enhanced Header */}
      <header 
        className="relative z-40 p-4 sm:p-6 shadow-2xl border-b transition-all duration-500 backdrop-blur-sm"
        style={{
          background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
          borderColor: 'var(--border-color)',
          boxShadow: '0 4px 20px var(--shadow-color)'
        }}
      >
        <div className="max-w-6xl mx-auto relative">
          {/* Enhanced Controls Row */}
          <div className="absolute top-0 right-0 flex items-center gap-3">
            <LanguageSelector disabled={isProcessing} />
            <ThemeToggle theme={theme} onToggle={onThemeToggle} />
          </div>

          {/* Magical Header Content */}
          <div className="text-center pr-32 sm:pr-40">
            <div className="flex items-center justify-center gap-3 sm:gap-4 mb-2 sm:mb-3">
              <div className="relative">
                <MapPin 
                  size={28} 
                  className="pulse-animation sm:w-10 sm:h-10 drop-shadow-lg" 
                  style={{ color: 'var(--accent-primary)' }}
                />
                <Sparkles size={12} className="absolute -top-1 -right-1 text-yellow-400 animate-pulse" />
              </div>
              <h1 
                className="text-3xl sm:text-5xl lg:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-500 via-red-500 to-pink-500 animate-gradient"
                style={{ 
                  fontFamily: 'Samarkan, serif',
                  textShadow: '3px 3px 6px var(--shadow-color)',
                  backgroundSize: '200% 200%'
                }}
              >
                Rahi.ai
              </h1>
              <div className="relative">
                <MapPin 
                  size={28} 
                  className="pulse-animation sm:w-10 sm:h-10 drop-shadow-lg" 
                  style={{ color: 'var(--accent-primary)' }}
                />
                <Star size={12} className="absolute -top-1 -left-1 text-blue-400 animate-pulse" />
              </div>
            </div>
            <p 
              className="text-base sm:text-xl font-semibold transition-colors duration-300 flex items-center justify-center gap-2"
              style={{ color: 'var(--text-secondary)' }}
            >
              <Zap size={16} className="animate-pulse" />
              Your magical AI companion for exploring India's wonders
              <Sparkles size={16} className="animate-pulse" />
            </p>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 flex flex-col p-3 sm:p-4 lg:p-6 pb-0">
        <div className="w-full max-w-5xl mx-auto flex-1 flex flex-col space-y-4 sm:space-y-6">
          
          {/* Enhanced About Section */}
          <div className="w-full hidden sm:block">
            <AboutSection />
          </div>

          {/* Enhanced Chat Container */}
          <div 
            className="flex-1 rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden transition-all duration-500 backdrop-blur-sm"
            style={{
              background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.05))`,
              border: '2px solid var(--border-color)',
              boxShadow: '0 12px 40px var(--shadow-color)',
              minHeight: '350px'
            }}
          >
            {/* Chat Messages */}
            <div 
              ref={chatBoxRef}
              className="h-full overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-5 custom-scrollbar"
              style={{
                background: `linear-gradient(to bottom, var(--bg-secondary), var(--bg-primary))`,
                paddingBottom: '1.5rem'
              }}
            >
              {messages.length === 0 ? (
                <div className={`flex items-center justify-center h-full ${welcomeShown ? 'fade-in-up' : 'opacity-0'}`}>
                  <div className="text-center px-4">
                    <div className="text-5xl sm:text-7xl mb-4 sm:mb-6">🏛️</div>
                    <div className="flex items-center justify-center gap-2 mb-3">
                      <Star size={20} className="text-yellow-400 animate-pulse" />
                      <p 
                        className="text-xl sm:text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-500 to-pink-500"
                      >
                        Namaste! Welcome to Rahi.ai
                      </p>
                      <Star size={20} className="text-yellow-400 animate-pulse" />
                    </div>
                    <p 
                      className="text-base sm:text-lg flex items-center justify-center gap-2"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      <Sparkles size={16} className="animate-pulse" />
                      Ask me about India's destinations, culture, festivals, or travel tips...
                      <Zap size={16} className="animate-pulse" />
                    </p>
                  </div>
                </div>
              ) : (
                messages.map(message => (
                  <ChatMessage 
                    key={message.id}
                    message={message}
                    autoPlay={autoPlay}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Enhanced Fixed Input Controls */}
      <div 
        className="sticky bottom-0 left-0 right-0 z-50 p-4 sm:p-5 transition-all duration-500 border-t backdrop-blur-md"
        style={{
          background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
          borderColor: 'var(--border-color)',
          boxShadow: '0 -6px 20px var(--shadow-color)'
        }}
      >
        <div className="max-w-5xl mx-auto space-y-4">
          {/* Main Input Row */}
          <div className="flex items-center gap-3 sm:gap-4">
            <MessageInput 
              onSendMessage={handleSendMessage} 
              disabled={inputsDisabled}
            />
            <VoiceInput 
              isRecording={isRecording}
              onRecordingChange={setIsRecording}
              onTranscription={handleVoiceTranscription}
              disabled={isProcessing}
            />
          </div>
          
          {/* Enhanced Controls Row */}
          <div className="flex items-center justify-center gap-6 text-sm sm:text-base">
            {/* Enhanced Auto-play toggle */}
            <button
              type="button"
              onClick={handleAutoPlayToggle}
              disabled={isProcessing}
              className="flex items-center gap-2 sm:gap-3 px-4 sm:px-6 py-3 rounded-full transition-all duration-300 hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 backdrop-blur-sm"
              style={{
                background: autoPlay 
                  ? `linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))` 
                  : 'var(--bg-primary)',
                color: autoPlay ? 'white' : 'var(--text-secondary)',
                border: `2px solid var(--border-color)`,
                boxShadow: '0 4px 12px var(--shadow-color)'
              }}
            >
              <span className="text-lg">{autoPlay ? '🔊' : '🔇'}</span>
              <span className="hidden sm:inline font-medium">Auto-play responses</span>
              <span className="sm:hidden font-medium">Auto-play</span>
              {autoPlay && <Sparkles size={16} className="animate-pulse" />}
            </button>
            
            {/* Enhanced Processing indicator */}
            {isProcessing && (
              <div className="flex items-center gap-2 sm:gap-3 px-4 py-2 rounded-full backdrop-blur-sm" 
                   style={{ 
                     backgroundColor: 'rgba(255,255,255,0.1)',
                     border: '1px solid var(--accent-primary)',
                     color: 'var(--accent-primary)' 
                   }}>
                <div 
                  className="animate-spin w-4 h-4 sm:w-5 sm:h-5 border-2 border-t-transparent rounded-full"
                  style={{ borderColor: 'var(--accent-primary)' }}
                ></div>
                <span className="text-sm sm:text-base font-medium">Processing magic...</span>
                <Zap size={16} className="animate-pulse" />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Footer */}
      <div className="hidden sm:flex items-center justify-center text-sm py-3 gap-2" style={{ color: 'var(--text-secondary)' }}>
        <Star size={16} className="text-yellow-400 animate-pulse" />
        <span>Powered by AI Magic • Experience the wonders of Incredible India</span>
        <span className="text-xl">🌟</span>
        <Star size={16} className="text-yellow-400 animate-pulse" />
      </div>
    </div>
  );
};

export default ChatBot;