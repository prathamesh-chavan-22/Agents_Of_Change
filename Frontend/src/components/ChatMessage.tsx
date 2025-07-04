import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';

interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  audioUrl?: string;
  isLoading?: boolean;
}

interface ChatMessageProps {
  message: Message;
  autoPlay: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, autoPlay }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    // Auto-play logic for bot messages with audio
    if (message.audioUrl && autoPlay && message.sender === 'bot' && !message.isLoading) {
      const timer = setTimeout(() => {
        handlePlayAudio();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [message.audioUrl, autoPlay, message.sender, message.isLoading]);

  useEffect(() => {
    // Cleanup audio when component unmounts
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
        audioRef.current = null;
      }
    };
  }, []);

  const handlePlayAudio = async () => {
    if (!message.audioUrl) {
      console.warn('No audio URL provided');
      return;
    }

    try {
      // If audio is currently playing, pause it
      if (audioRef.current && !audioRef.current.paused) {
        audioRef.current.pause();
        setIsPlaying(false);
        return;
      }

      setIsLoading(true);
      setHasError(false);

      // Create new audio element if needed
      if (!audioRef.current) {
        audioRef.current = new Audio();
        
        audioRef.current.onloadstart = () => {
          console.log('Audio loading started');
          setIsLoading(true);
        };
        
        audioRef.current.oncanplay = () => {
          console.log('Audio can play');
          setIsLoading(false);
        };
        
        audioRef.current.onplay = () => {
          console.log('Audio started playing');
          setIsPlaying(true);
          setIsLoading(false);
        };
        
        audioRef.current.onpause = () => {
          console.log('Audio paused');
          setIsPlaying(false);
        };
        
        audioRef.current.onended = () => {
          console.log('Audio ended');
          setIsPlaying(false);
        };
        
        audioRef.current.onerror = (e) => {
          console.error('Audio error:', e);
          setHasError(true);
          setIsPlaying(false);
          setIsLoading(false);
        };
      }

      // Set source and play
      if (audioRef.current.src !== message.audioUrl) {
        audioRef.current.src = message.audioUrl;
      }

      await audioRef.current.play();
      
    } catch (error) {
      console.error('Error playing audio:', error);
      setHasError(true);
      setIsPlaying(false);
      setIsLoading(false);
      
      // Show user-friendly error message
      if (error instanceof DOMException) {
        if (error.name === 'NotAllowedError') {
          alert('Audio playback blocked. Please enable audio autoplay or click play manually.');
        } else if (error.name === 'NotSupportedError') {
          alert('Audio format not supported by your browser.');
        } else {
          alert('Unable to play audio. Please try again.');
        }
      }
    }
  };

  if (message.sender === 'user') {
    return (
      <div className="flex justify-end fade-in-up">
        <div 
          className="max-w-xs sm:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl rounded-br-sm shadow-lg transition-all duration-300"
          style={{
            background: `linear-gradient(135deg, var(--accent-secondary), var(--accent-primary))`,
            color: 'white'
          }}
        >
          <p className="text-sm sm:text-base break-words">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start fade-in-up">
      <div className="max-w-xs sm:max-w-md lg:max-w-lg">
        <div 
          className="px-4 py-3 rounded-2xl rounded-bl-sm shadow-lg border transition-all duration-300"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderColor: 'var(--border-color)',
            boxShadow: '0 2px 8px var(--shadow-color)'
          }}
        >
          {message.isLoading ? (
            <div className="flex items-center gap-2" style={{ color: 'var(--accent-primary)' }}>
              <div 
                className="animate-spin w-4 h-4 border-2 border-t-transparent rounded-full"
                style={{ borderColor: 'var(--accent-primary)' }}
              ></div>
              <span className="text-sm">Thinking...</span>
            </div>
          ) : (
            <>
              <p 
                className="text-sm sm:text-base leading-relaxed break-words"
                style={{ color: 'var(--text-primary)' }}
              >
                {message.content}
              </p>
              
              {message.audioUrl && (
                <div 
                  className="mt-3 pt-3 border-t"
                  style={{ borderColor: 'var(--border-color)' }}
                >
                  <button
                    type="button"
                    onClick={handlePlayAudio}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all duration-200 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-50"
                    style={{
                      backgroundColor: hasError ? '#dc2626' : 'var(--accent-primary)',
                      color: 'white'
                    }}
                  >
                    {isLoading ? (
                      <>
                        <div className="animate-spin w-4 h-4 border-2 border-t-transparent rounded-full border-white"></div>
                        <span>Loading...</span>
                      </>
                    ) : hasError ? (
                      <>
                        <VolumeX size={16} />
                        <span>Audio Error</span>
                      </>
                    ) : isPlaying ? (
                      <>
                        <Pause size={16} />
                        <span>Pause</span>
                      </>
                    ) : (
                      <>
                        <Play size={16} />
                        <span>Play Audio</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;