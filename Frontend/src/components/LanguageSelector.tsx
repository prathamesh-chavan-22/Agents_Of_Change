import React, { useState } from 'react';
import { Globe, Check, Sparkles, Star } from 'lucide-react';

interface LanguageSelectorProps {
  disabled?: boolean;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({ disabled = false }) => {
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);

  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸', nativeName: 'English' },
    { code: 'hi', name: 'Hindi', flag: '🇮🇳', nativeName: 'हिंदी' },
    { code: 'mr', name: 'Marathi', flag: '🇮🇳', nativeName: 'मराठी' },
    { code: 'ta', name: 'Tamil', flag: '🇮🇳', nativeName: 'தமிழ்' },
    { code: 'bn', name: 'Bengali', flag: '🇮🇳', nativeName: 'বাংলা' },
    { code: 'or', name: 'Odia', flag: '🇮🇳', nativeName: 'ଓଡ଼ିଆ' }

  ];

  const apiBase = "http://localhost:8000";

  const setLanguage = async (languageCode: string) => {
    if (disabled || isLoading) return;

    setIsLoading(true);
    setStatusMessage('');

    try {
      console.log('Setting language to:', languageCode);
      
      const response = await fetch(`${apiBase}/set_language`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ language: languageCode })
      });

      console.log('Language API response status:', response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Language API response:', data);
      
      setSelectedLanguage(languageCode);
      setStatusMessage(data.message || `Language set to ${languages.find(l => l.code === languageCode)?.name}`);
      setShowDropdown(false);
      
      // Clear status message after 3 seconds
      setTimeout(() => setStatusMessage(''), 3000);
      
    } catch (error) {
      console.error('Error setting language:', error);
      
      let errorMessage = 'Failed to set language: ';
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage += 'Please check if the server is running on http://localhost:8000';
      } else if (error instanceof Error) {
        errorMessage += error.message;
      } else {
        errorMessage += 'Please try again.';
      }
      
      setStatusMessage(errorMessage);
      
      // Clear error message after 5 seconds
      setTimeout(() => setStatusMessage(''), 5000);
    } finally {
      setIsLoading(false);
    }
  };

  const currentLanguage = languages.find(l => l.code === selectedLanguage);

  return (
    <div className="relative">
      {/* Language Selector Button */}
      <button
        type="button"
        onClick={() => !disabled && !isLoading && setShowDropdown(!showDropdown)}
        disabled={disabled || isLoading}
        className="flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 backdrop-blur-sm"
        style={{
          background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
          border: '2px solid var(--border-color)',
          color: 'var(--text-primary)',
          boxShadow: '0 4px 16px var(--shadow-color)'
        }}
      >
        <div className="relative">
          <Globe size={20} className="drop-shadow-sm" />
          <Sparkles size={8} className="absolute -top-1 -right-1 text-yellow-400 animate-pulse" />
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-lg">{currentLanguage?.flag}</span>
          <span className="font-medium hidden sm:inline">{currentLanguage?.name}</span>
          <span className="font-medium sm:hidden">{currentLanguage?.code.toUpperCase()}</span>
        </div>
        
        {isLoading ? (
          <div 
            className="animate-spin w-4 h-4 border-2 border-t-transparent rounded-full"
            style={{ borderColor: 'var(--accent-primary)' }}
          />
        ) : (
          <div className={`transform transition-transform duration-300 ${showDropdown ? 'rotate-180' : ''}`}>
            <Star size={16} className="text-yellow-400" />
          </div>
        )}
      </button>

      {/* Dropdown Menu */}
      {showDropdown && !disabled && !isLoading && (
        <div 
          className="absolute top-full left-0 right-0 mt-2 rounded-xl shadow-2xl border-2 overflow-hidden z-50 backdrop-blur-sm"
          style={{
            background: `linear-gradient(135deg, var(--bg-secondary), rgba(255,255,255,0.1))`,
            borderColor: 'var(--border-color)',
            boxShadow: '0 12px 40px var(--shadow-color)'
          }}
        >
          {languages.map((language) => (
            <button
              key={language.code}
              onClick={() => setLanguage(language.code)}
              className="w-full flex items-center gap-3 px-4 py-3 transition-all duration-200 hover:scale-105 first:rounded-t-lg last:rounded-b-lg"
              style={{
                backgroundColor: selectedLanguage === language.code 
                  ? 'var(--accent-primary)' 
                  : 'transparent',
                color: selectedLanguage === language.code 
                  ? 'white' 
                  : 'var(--text-primary)'
              }}
            >
              <span className="text-lg">{language.flag}</span>
              <div className="flex-1 text-left">
                <div className="font-medium">{language.name}</div>
                <div className="text-sm opacity-75">{language.nativeName}</div>
              </div>
              {selectedLanguage === language.code && (
                <Check size={16} className="drop-shadow-sm" />
              )}
            </button>
          ))}
        </div>
      )}

      {/* Status Message */}
      {statusMessage && (
        <div 
          className="absolute top-full left-0 right-0 mt-2 p-3 rounded-xl border-2 text-sm font-medium backdrop-blur-sm"
          style={{
            background: statusMessage.includes('Failed') || statusMessage.includes('Error')
              ? 'linear-gradient(135deg, #dc2626, #ef4444)'
              : 'linear-gradient(135deg, var(--accent-tertiary), #4ade80)',
            borderColor: statusMessage.includes('Failed') || statusMessage.includes('Error')
              ? '#dc2626'
              : 'var(--accent-tertiary)',
            color: 'white',
            boxShadow: '0 4px 16px var(--shadow-color)',
            zIndex: 60
          }}
        >
          <div className="flex items-center gap-2">
            {statusMessage.includes('Failed') || statusMessage.includes('Error') ? (
              <span>❌</span>
            ) : (
              <Sparkles size={16} className="animate-pulse" />
            )}
            <span>{statusMessage}</span>
          </div>
        </div>
      )}

      {/* Click outside to close dropdown */}
      {showDropdown && (
        <div 
          className="fixed inset-0 z-40"
          onClick={() => setShowDropdown(false)}
        />
      )}
    </div>
  );
};

export default LanguageSelector;