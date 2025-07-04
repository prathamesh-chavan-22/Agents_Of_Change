import React from 'react';
import { Sun, Moon, Sparkles } from 'lucide-react';

interface ThemeToggleProps {
  theme: 'light' | 'dark';
  onToggle: () => void;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ theme, onToggle }) => {
  return (
    <button
      onClick={onToggle}
      className="relative p-4 rounded-full shadow-2xl transition-all duration-500 hover:scale-110 active:scale-95 backdrop-blur-sm overflow-hidden group"
      style={{
        background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
        color: 'var(--accent-primary)',
        border: '2px solid var(--border-color)',
        boxShadow: '0 8px 24px var(--shadow-color)'
      }}
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      {/* Magical sparkle overlay */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <Sparkles size={12} className="absolute top-1 right-1 text-yellow-400 animate-pulse" />
        <Sparkles size={10} className="absolute bottom-1 left-1 text-blue-400 animate-pulse" />
      </div>
      
      <div className="relative z-10 transition-transform duration-500">
        {theme === 'light' ? (
          <Moon size={24} className="transition-all duration-500 hover:rotate-12 drop-shadow-lg" />
        ) : (
          <Sun size={24} className="transition-all duration-500 hover:rotate-12 drop-shadow-lg" />
        )}
      </div>
      
      {/* Magical glow effect */}
      <div 
        className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-300"
        style={{
          background: `radial-gradient(circle, var(--accent-primary), transparent)`
        }}
      />
    </button>
  );
};

export default ThemeToggle;