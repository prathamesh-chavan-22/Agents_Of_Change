import React, { useState } from 'react';
import { Info, ChevronRight, Sparkles, Star, Zap } from 'lucide-react';

const AboutSection: React.FC = () => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div 
      className="rounded-2xl shadow-2xl overflow-hidden transition-all duration-500 hover:scale-105 backdrop-blur-sm"
      style={{
        background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
        border: '2px solid var(--border-color)',
        boxShadow: '0 8px 32px var(--shadow-color)'
      }}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-5 font-bold text-left flex items-center justify-between transition-all duration-300 hover:opacity-90 relative overflow-hidden"
        style={{
          background: `linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))`,
          color: 'white'
        }}
      >
        {/* Magical sparkle overlay */}
        <div className="absolute inset-0 opacity-20">
          <Sparkles size={20} className="absolute top-2 right-8 animate-pulse" />
          <Star size={16} className="absolute bottom-2 left-8 animate-pulse" />
          <Zap size={18} className="absolute top-3 left-1/3 animate-pulse" />
        </div>
        
        <div className="flex items-center gap-4 relative z-10">
          <div className="relative">
            <Info size={24} className="drop-shadow-lg" />
            <Sparkles size={10} className="absolute -top-1 -right-1 animate-pulse" />
          </div>
          <div>
            <span className="hidden sm:inline text-lg">About Your Magical India Travel Assistant</span>
            <span className="sm:hidden text-lg">About Rahi.ai ✨</span>
          </div>
        </div>
        <ChevronRight 
          size={24} 
          className={`transform transition-transform duration-300 drop-shadow-lg ${isExpanded ? 'rotate-90' : ''}`}
        />
      </button>
      
      {isExpanded && (
        <div 
          className="p-8 transition-all duration-500 backdrop-blur-sm"
          style={{
            background: `linear-gradient(135deg, var(--bg-primary), var(--bg-secondary))`,
            borderLeft: `4px solid var(--accent-tertiary)`
          }}
        >
          <div className="prose prose-sm max-w-none">
            <div className="flex items-start gap-3 mb-6">
              <Star size={24} className="text-yellow-400 animate-pulse mt-1" />
              <p 
                className="leading-relaxed text-lg"
                style={{ color: 'var(--text-primary)' }}
              >
                This <strong>AI-powered magical assistant</strong> helps travelers explore <strong>India's incredible destinations, rich culture, and heritage sites</strong> by providing real-time information using advanced AI and Retrieval-Augmented Generation (RAG) technology.
              </p>
            </div>
            
            <div className="flex items-start gap-3 mb-6">
              <Zap size={24} className="text-green-400 animate-pulse mt-1" />
              <p 
                className="leading-relaxed text-lg"
                style={{ color: 'var(--text-primary)' }}
              >
                🏛️ Get enchanting information about <strong>monuments, temples, festivals, local cuisine, and travel tips</strong> with accurate, spoken responses—your personal digital guide with a touch of magic!
              </p>
            </div>
            
            <div 
              className="rounded-xl p-6 border-2 transition-all duration-300 hover:scale-105 backdrop-blur-sm"
              style={{
                background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
                borderColor: 'var(--accent-primary)',
                boxShadow: '0 4px 16px var(--shadow-color)'
              }}
            >
              <div className="flex items-center gap-3 mb-4">
                <Sparkles size={24} className="text-purple-400 animate-pulse" />
                <p 
                  className="font-bold text-xl"
                  style={{ color: 'var(--text-primary)' }}
                >
                  How to use this magical experience:
                </p>
              </div>
              <ul className="space-y-3 text-base" style={{ color: 'var(--text-secondary)' }}>
                <li className="flex items-start gap-3 p-2 rounded-lg transition-colors duration-300" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}>
                  <span style={{ color: 'var(--accent-secondary)' }}>💬</span>
                  <span>Type your question about India and press <strong>Enter</strong> or click <strong>Send</strong> to begin the magic</span>
                </li>
                <li className="flex items-start gap-3 p-2 rounded-lg transition-colors duration-300" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}>
                  <span style={{ color: 'var(--accent-tertiary)' }}>🎤</span>
                  <span>Click the microphone button to use voice input—speak naturally and experience instant transcription magic</span>
                </li>
                <li className="flex items-start gap-3 p-2 rounded-lg transition-colors duration-300" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}>
                  <span style={{ color: 'var(--accent-primary)' }}>🔊</span>
                  <span>Enable auto-play to hear responses automatically, or control the magical audio playback manually</span>
                </li>
                <li className="flex items-start gap-3 p-2 rounded-lg transition-colors duration-300" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}>
                  <span style={{ color: 'var(--accent-secondary)' }}>🌟</span>
                  <span>Ask about destinations, culture, food, festivals, travel tips, and discover India's hidden magical treasures!</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AboutSection;