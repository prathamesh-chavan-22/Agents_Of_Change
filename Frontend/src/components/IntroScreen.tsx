import React, { useState, useEffect } from 'react';
import { MapPin, Sparkles, Globe, Heart, ArrowRight, Compass, Star, Zap } from 'lucide-react';

interface IntroScreenProps {
  theme: 'light' | 'dark';
  onComplete: () => void;
}

const IntroScreen: React.FC<IntroScreenProps> = ({ theme, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(true);
  const [showParticles, setShowParticles] = useState(false);

  const steps = [
    {
      icon: <MapPin size={56} className="text-orange-500 drop-shadow-lg" />,
      title: "Welcome to Rahi.ai",
      subtitle: "Your AI-powered travel companion for India",
      description: "Discover the incredible diversity, rich culture, and timeless heritage of India through intelligent conversations.",
      gradient: "from-orange-500 via-red-500 to-pink-500"
    },
    {
      icon: <Globe size={56} className="text-blue-500 drop-shadow-lg" />,
      title: "Explore India's Wonders",
      subtitle: "From ancient monuments to vibrant festivals",
      description: "Get personalized recommendations for destinations, local cuisine, hidden gems, and cultural experiences across the subcontinent.",
      gradient: "from-blue-500 via-purple-500 to-indigo-500"
    },
    {
      icon: <Sparkles size={56} className="text-green-500 drop-shadow-lg" />,
      title: "AI-Powered Magic",
      subtitle: "Voice and text interactions with intelligence",
      description: "Ask questions naturally using voice or text, and receive detailed responses with immersive audio playback for an enchanting experience.",
      gradient: "from-green-500 via-emerald-500 to-teal-500"
    },
    {
      icon: <Heart size={56} className="text-red-500 drop-shadow-lg animate-pulse" />,
      title: "Ready for Adventure?",
      subtitle: "Start your magical journey through Incredible India",
      description: "Let's explore the magic, mystery, and magnificence of India together. Your extraordinary adventure awaits! 🇮🇳",
      gradient: "from-red-500 via-pink-500 to-rose-500"
    }
  ];

  useEffect(() => {
    // Show particles after initial load
    const timer = setTimeout(() => {
      setShowParticles(true);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
      }, 3500);
      return () => clearTimeout(timer);
    }
  }, [currentStep, steps.length]);

  const handleGetStarted = () => {
    setIsVisible(false);
    setTimeout(() => {
      onComplete();
    }, 800);
  };

  const handleSkip = () => {
    setCurrentStep(steps.length - 1);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div 
      className={`fixed inset-0 z-50 flex items-center justify-center transition-all duration-800 overflow-hidden ${
        isVisible ? 'opacity-100' : 'opacity-0'
      }`}
      style={{ 
        background: `linear-gradient(135deg, var(--bg-primary), var(--bg-secondary))`,
      }}
    >
      {/* Magical Floating Particles */}
      {showParticles && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${3 + Math.random() * 4}s`
              }}
            >
              {i % 4 === 0 ? (
                <Star size={12} className="text-yellow-400 opacity-60" />
              ) : i % 4 === 1 ? (
                <Sparkles size={10} className="text-blue-400 opacity-50" />
              ) : i % 4 === 2 ? (
                <Zap size={8} className="text-green-400 opacity-40" />
              ) : (
                <Compass size={14} className="text-purple-400 opacity-55" />
              )}
            </div>
          ))}
        </div>
      )}

      {/* Animated Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div 
          className="absolute inset-0 animate-pulse"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='120' height='120' viewBox='0 0 120 120' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23e65100' fill-opacity='0.15'%3E%3Cpath d='M60 60c0-22.091-17.909-40-40-40s-40 17.909-40 40 17.909 40 40 40 40-17.909 40-40zm0 0c0 22.091 17.909 40 40 40s40-17.909 40-40-17.909-40-40-40-40 17.909-40 40z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }} 
        />
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-3xl mx-auto px-6 text-center">
        {/* Magical Logo Section */}
        <div className="mb-12 fade-in-up">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="relative">
              <MapPin 
                size={48} 
                className="pulse-animation drop-shadow-2xl" 
                style={{ color: 'var(--accent-primary)' }}
              />
              <div className="absolute -top-1 -right-1">
                <Sparkles size={16} className="text-yellow-400 animate-pulse" />
              </div>
            </div>
            <h1 
              className="text-6xl sm:text-7xl lg:text-8xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-500 via-red-500 to-pink-500 animate-gradient"
              style={{ 
                fontFamily: 'Samarkan, serif',
                textShadow: '4px 4px 8px var(--shadow-color)',
                backgroundSize: '200% 200%'
              }}
            >
              Rahi.ai
            </h1>
            <div className="relative">
              <MapPin 
                size={48} 
                className="pulse-animation drop-shadow-2xl" 
                style={{ color: 'var(--accent-primary)' }}
              />
              <div className="absolute -top-1 -left-1">
                <Star size={16} className="text-blue-400 animate-pulse" />
              </div>
            </div>
          </div>
          <div className="flex items-center justify-center gap-2 text-lg font-medium" style={{ color: 'var(--text-secondary)' }}>
            <Compass size={20} className="animate-spin-slow" />
            <span>Your Magical Gateway to Incredible India</span>
            <Compass size={20} className="animate-spin-slow" style={{ animationDirection: 'reverse' }} />
          </div>
        </div>

        {/* Enhanced Step Content */}
        <div 
          key={currentStep}
          className="fade-in-up mb-12 p-10 rounded-3xl shadow-2xl transition-all duration-500 transform hover:scale-105 backdrop-blur-sm"
          style={{
            background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
            border: '3px solid transparent',
            backgroundClip: 'padding-box',
            boxShadow: '0 20px 60px var(--shadow-color), inset 0 1px 0 rgba(255,255,255,0.2)'
          }}
        >
          {/* Gradient Border Effect */}
          <div 
            className="absolute inset-0 rounded-3xl opacity-75 blur-sm"
            style={{
              background: `linear-gradient(135deg, ${steps[currentStep].gradient.split(' ').join(', ')})`,
              zIndex: -1
            }}
          />

          {/* Floating Icon */}
          <div className="flex justify-center mb-8">
            <div 
              className="p-6 rounded-full transition-all duration-500 transform hover:rotate-12 hover:scale-110 animate-float"
              style={{
                background: `linear-gradient(135deg, var(--bg-primary), var(--bg-secondary))`,
                border: '3px solid var(--border-color)',
                boxShadow: '0 10px 30px var(--shadow-color)'
              }}
            >
              {steps[currentStep].icon}
            </div>
          </div>

          {/* Enhanced Typography */}
          <h2 
            className="text-4xl sm:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r"
            style={{ 
              backgroundImage: `linear-gradient(135deg, ${steps[currentStep].gradient.split(' ').join(', ')})`,
              textShadow: '2px 2px 4px var(--shadow-color)'
            }}
          >
            {steps[currentStep].title}
          </h2>

          <p 
            className="text-xl sm:text-2xl font-semibold mb-6"
            style={{ color: 'var(--accent-primary)' }}
          >
            {steps[currentStep].subtitle}
          </p>

          <p 
            className="text-lg sm:text-xl leading-relaxed max-w-2xl mx-auto"
            style={{ color: 'var(--text-secondary)' }}
          >
            {steps[currentStep].description}
          </p>
        </div>

        {/* Enhanced Progress Indicators */}
        <div className="flex justify-center gap-3 mb-12">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`rounded-full transition-all duration-500 ${
                index === currentStep ? 'scale-150 shadow-lg' : 'scale-100'
              }`}
              style={{
                width: index === currentStep ? '16px' : '12px',
                height: index === currentStep ? '16px' : '12px',
                background: index <= currentStep 
                  ? `linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))` 
                  : 'var(--border-color)',
                boxShadow: index === currentStep ? '0 4px 12px var(--accent-primary)' : 'none'
              }}
            />
          ))}
        </div>

        {/* Enhanced Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
          {currentStep < steps.length - 1 ? (
            <>
              <button
                onClick={handleSkip}
                className="px-8 py-4 rounded-2xl font-medium transition-all duration-300 hover:scale-105 active:scale-95 backdrop-blur-sm"
                style={{
                  backgroundColor: 'rgba(255,255,255,0.1)',
                  color: 'var(--text-secondary)',
                  border: '2px solid var(--border-color)',
                  boxShadow: '0 4px 16px var(--shadow-color)'
                }}
              >
                Skip the Magic ✨
              </button>
              <button
                onClick={() => setCurrentStep(prev => prev + 1)}
                className="px-12 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 active:scale-95 flex items-center gap-3 shadow-2xl"
                style={{
                  background: `linear-gradient(135deg, var(--accent-secondary), var(--accent-primary))`,
                  color: 'white',
                  boxShadow: '0 8px 32px var(--shadow-color)'
                }}
              >
                <Sparkles size={24} className="animate-pulse" />
                <span>Continue Journey</span>
                <ArrowRight size={24} />
              </button>
            </>
          ) : (
            <button
              onClick={handleGetStarted}
              className="px-16 py-6 rounded-2xl text-xl font-bold transition-all duration-300 hover:scale-110 active:scale-95 flex items-center gap-4 shadow-2xl animate-pulse"
              style={{
                background: `linear-gradient(135deg, var(--accent-secondary), var(--accent-primary), var(--accent-tertiary))`,
                backgroundSize: '200% 200%',
                color: 'white',
                boxShadow: '0 12px 40px var(--shadow-color)',
                animation: 'gradientShift 3s ease infinite, pulse 2s infinite'
              }}
            >
              <Heart size={28} className="animate-pulse" />
              <span>Begin Magical Journey</span>
              <Sparkles size={28} className="animate-pulse" />
            </button>
          )}
        </div>

        {/* Enhanced Footer */}
        <div className="mt-12 text-base flex items-center justify-center gap-2" style={{ color: 'var(--text-secondary)' }}>
          <Star size={20} className="text-yellow-400 animate-pulse" />
          <span>Powered by AI Magic • Experience Incredible India</span>
          <span className="text-2xl">🇮🇳</span>
          <Star size={20} className="text-yellow-400 animate-pulse" />
        </div>
      </div>
    </div>
  );
};

export default IntroScreen;