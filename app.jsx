import React, { useState, useRef, useEffect } from 'react';

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { text: "Welcome to SmartWeather CLI \uD83C\uDF26\uFE0F\nType /help to see available commands.", isUser: false },
  ]);
  const [darkMode, setDarkMode] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    const newMessage = { text: input, isUser: true };
    setMessages((prev) => [...prev, newMessage]);
    parseCommand(input);
    setInput('');
  };

  const parseCommand = (cmd) => {
    const lower = cmd.toLowerCase().trim();

    const mockWeather = {
      city: "Delhi",
      temp: 28,
      feelsLike: 30,
      humidity: 65,
      windSpeed: 10,
      description: "Partly cloudy",
      icon: "\uD83C\uDF24\uFE0F",
      forecast: [
        { day: "Mon", temp: 29, icon: "\u2600\uFE0F" },
        { day: "Tue", temp: 31, icon: "\u26C5" },
        { day: "Wed", temp: 27, icon: "\uD83C\uDF27\uFE0F" },
        { day: "Thu", temp: 26, icon: "\u2601\uFE0F" },
        { day: "Fri", temp: 25, icon: "\uD83C\uDF26\uFE0F" },
        { day: "Sat", temp: 24, icon: "\uD83C\uDF27\uFE0F" },
        { day: "Sun", temp: 23, icon: "\u26C8\uFE0F" }
      ],
      hourly: Array.from({ length: 24 }, (_, i) => ({ hour: `${i}:00`, temp: 28 + Math.sin(i / 5) * 2 })),
      sunrise: "05:42 AM",
      sunset: "07:12 PM",
      aqi: 90,
      alert: "Heatwave Alert in Delhi today!",
      tips: ["Drink more water", "Avoid direct sun exposure", "Use sunscreen"]
    };

    let response = "";
    if (lower.startsWith("/weather")) {
      const location = lower.split(" ")[1] || "Delhi";
      response = (
        <div className="space-y-2">
          <h3 className="font-bold">{mockWeather.icon} Current Weather in {location}</h3>
          <p>\uD83C\uDF21\uFE0F Temperature: {mockWeather.temp}\u00B0C</p>
          <p>\uD83D\uDCA7 Humidity: {mockWeather.humidity}%</p>
          <p>\uD83C\uDF2C\uFE0F Wind: {mockWeather.windSpeed} km/h</p>
          <p>\uD83D\uDCDD Description: {mockWeather.description}</p>
        </div>
      );
    } else if (lower.startsWith("/forecast")) {
      const location = lower.split(" ")[1] || "Delhi";
      response = (
        <div>
          <h3 className="font-bold">\uD83D\uDCC5 7-Day Forecast in {location}</h3>
          <div className="flex gap-2 mt-2">
            {mockWeather.forecast.map((day, i) => (
              <div key={i} className="bg-gray-800 p-2 rounded text-center w-16">
                <span>{day.day}</span><br />{day.icon}<br />{day.temp}\u00B0C
              </div>
            ))}
          </div>
        </div>
      );
    } else if (lower.startsWith("/hourly")) {
      const location = lower.split(" ")[1] || "Delhi";
      response = (
        <div>
          <h3 className="font-bold">\u23F1\uFE0F Hourly Forecast in {location}</h3>
          <div className="mt-2 max-w-md h-40">
            <svg width="100%" height="100%" viewBox="0 0 500 100">
              {mockWeather.hourly.map((h, i) => (
                <rect key={i} x={i * 20} y={100 - h.temp * 2} width="10" height={h.temp * 2} fill="#10B981" />
              ))}
            </svg>
          </div>
        </div>
      );
    } else if (lower.startsWith("/aqi")) {
      response = (
        <div>
          <h3 className="font-bold">\uD83C\uDF00 Air Quality Index (AQI)</h3>
          <p>AQI in Delhi: {mockWeather.aqi} (Moderate)</p>
        </div>
      );
    } else if (lower.startsWith("/sunrise")) {
      const location = lower.split(" ")[1] || "Delhi";
      response = (
        <div>
          <h3 className="font-bold">\uD83C\uDF07 Sunrise & Sunset in {location}</h3>
          <p>Sunrise: {mockWeather.sunrise}</p>
          <p>Sunset: {mockWeather.sunset}</p>
        </div>
      );
    } else if (lower === "/alerts") {
      response = (
        <div className="text-red-400">
          <h3 className="font-bold">\uD83D\uDEA8 Weather Alerts</h3>
          <p>{mockWeather.alert}</p>
        </div>
      );
    } else if (lower === "/tips") {
      response = (
        <div>
          <h3 className="font-bold">\uD83E\uDE20 Smart Suggestions</h3>
          <ul className="list-disc ml-5">
            {mockWeather.tips.map((tip, i) => <li key={i}>{tip}</li>)}
          </ul>
        </div>
      );
    } else if (lower.startsWith("/chart")) {
      response = (
        <div>
          <h3 className="font-bold">\uD83D\uDCCA Temperature Chart</h3>
          <div className="mt-2 max-w-md h-40">
            <svg width="100%" height="100%" viewBox="0 0 500 100">
              {mockWeather.hourly.map((h, i) => (
                <circle key={i} cx={i * 20} cy={100 - h.temp * 2} r="3" fill="#fbbf24" />
              ))}
            </svg>
          </div>
        </div>
      );
    } else if (lower === "/here") {
      response = "\uD83D\uDCCD Detecting your location...";
    } else if (lower.startsWith("/export")) {
      const parts = lower.split(" ");
      const location = parts[1];
      const format = parts[2];
      response = `\uD83D\uDCC4 Exporting weather report for ${location} as ${format.toUpperCase()}...`;
    } else if (lower === "/help") {
      response = (
        <div>
          <h3 className="font-bold">\uD83D\uDCDC Available Commands:</h3>
          <ul className="list-disc ml-5 space-y-1 text-sm">
            <li><code>/weather [location]</code> – Show current weather</li>
            <li><code>/forecast [location]</code> – 7-day forecast</li>
            <li><code>/aqi [location]</code> – Air Quality Index</li>
            <li><code>/hourly [location]</code> – Hourly forecast</li>
            <li><code>/sunrise [location]</code> – Sunrise & Sunset times</li>
            <li><code>/alerts</code> – Show active weather alerts</li>
            <li><code>/tips</code> – Smart suggestions</li>
            <li><code>/chart [location]</code> – Show temperature graph</li>
            <li><code>/export [location] [format]</code> – Export as PDF/CSV</li>
            <li><code>/here</code> – Detect location automatically</li>
            <li><code>/help</code> – Show this help menu</li>
          </ul>
        </div>
      );
    } else {
      response = "\u274C Unknown command. Type `/help` for list.";
    }

    setTimeout(() => {
      setMessages((prev) => [...prev, { text: response, isUser: false }]);
    }, 500);
  };

  const startVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert("Voice recognition not supported in this browser.");
      return;
    }
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
    };
    recognition.onerror = (event) => {
      alert("Voice recognition error: " + event.error);
    };
    recognition.start();
  };

  return (
    <div className={`min-h-screen flex flex-col ${darkMode ? 'bg-gray-900 text-green-400' : 'bg-gray-100 text-gray-800'} transition-colors duration-300`}>
      <header className="flex justify-between items-center px-4 py-2 border-b border-gray-700">
        <h1 className="text-lg font-bold">SmartWeather CLI \uD83C\uDF26\uFE0F</h1>
        <button onClick={() => setDarkMode(!darkMode)} className="text-sm underline">
          {darkMode ? "Light Mode" : "Dark Mode"}
        </button>
      </header>

      <div className="flex-grow overflow-y-auto p-4 space-y-2">
        {messages.map((msg, i) => (
          <div key={i} className={`mb-2 ${msg.isUser ? 'text-right' : ''}`}>
            <div className={`inline-block p-3 rounded-lg ${msg.isUser ? 'bg-green-900' : 'bg-gray-800'} max-w-xs sm:max-w-md md:max-w-lg`}>
              {typeof msg.text === 'string' ? <span dangerouslySetInnerHTML={{ __html: msg.text }} /> : msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-gray-700 p-4 flex items-center gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="/help"
          className={`flex-grow bg-transparent outline-none`}
        />
        <button onClick={startVoiceInput} className="px-3 py-1 bg-purple-600 hover:bg-purple-500 rounded">\uD83C\uDF99\uFE0F</button>
        <button onClick={handleSend} className="px-4 py-1 bg-green-600 hover:bg-green-500 rounded">Send</button>
      </div>
    </div>
  );
}
