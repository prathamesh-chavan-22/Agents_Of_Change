import React, { useState, useEffect } from 'react';
import ChatBot from './components/ChatBot';
import IntroScreen from './components/IntroScreen';

function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [showIntro, setShowIntro] = useState(true);

  useEffect(() => {
    // Load theme from localStorage
    const savedTheme = localStorage.getItem('rahi-theme') as 'light' | 'dark' | null;
    if (savedTheme) {
      setTheme(savedTheme);
    } else {
      // Check system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setTheme(prefersDark ? 'dark' : 'light');
    }

    // Always show intro on page refresh - magical experience every time!
    setShowIntro(true);
  }, []);

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('rahi-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const handleIntroComplete = () => {
    setShowIntro(false);
  };

  if (showIntro) {
    return <IntroScreen theme={theme} onComplete={handleIntroComplete} />;
  }

  return (
    <ChatBot theme={theme} onThemeToggle={toggleTheme} />
  );
}

export default App;