import React, { useState } from 'react';
import axios from 'axios'; 
import '../styles/toolsStyles.scss';

const Tools = ({ showTools, setShowTools }) => {

  const [activeTools, setActiveTools] = useState({
    practiceTool: false,
    historyTool: false,
    brainTool: false,
    analyticsTool: false,
    comingSoonTool: false
  });

  const handleToolClick = async (toolName, apiEndpoint) => {
    if (activeTools[toolName]) return;
    
    try {
      setActiveTools(prev => ({ ...prev, [toolName]: true }));
      
      if (apiEndpoint) {
        const response = await axios.post(`http://localhost:3001/${apiEndpoint}`);
        console.log(`Response from server:`, response.data.message);
      }
      
      setTimeout(() => {
        setActiveTools(prev => ({ ...prev, [toolName]: false }));
      }, 10000);
    } catch (error) {
      console.error(`Error launching ${toolName}:`, error.response ? error.response.data : error.message);
      setActiveTools(prev => ({ ...prev, [toolName]: false }));
    }
  };
  
  const handlePracticeTradingClick = async () => {
    try {
      const response = await axios.post('http://localhost:3001/launch-trading-app');
      console.log("Response from server:", response.data.message); 
    } catch (error) {
      console.error("Error launching trading app:", error.response ? error.response.data : error.message);
    }
  };
  
  const handleHistoryTradingClick = async () => {
    try {
      const response = await axios.post('http://localhost:3001/launch-history-app');
      console.log("Response from server:", response.data.message); 
    } catch (error) {
      console.error("Error launching history app:", error.response ? error.response.data : error.message);
    }
  };
  
  const handleBrainClick = async () => {
    try {
      const response = await axios.post('http://localhost:3001/launch-brain-app');
      console.log("Response from server:", response.data.message); 
    } catch (error) {
      console.error("Error launching history app:", error.response ? error.response.data : error.message);
    }
  };

  return (
    <div className={`toolsContainer ${showTools ? 'show' : 'hide'}`}>
      <div className="toolsHeader">
        <div className="headerDots">
          
        </div>
        <h2 className="mainToolsTitle">TRADING TOOLS</h2>
        <button className="closeButton" onClick={() => setShowTools(false)}></button>
      </div>
      <div className="toolsContent">
        <button 
          className={`toolBox ${activeTools.practiceTool ? 'active' : ''}`} 
          onClick={() => handleToolClick('practiceTool', 'launch-trading-app')}
          disabled={activeTools.practiceTool}
        >
          <div className="toolIconBox">
            <span className="toolIcon"></span>
          </div>
          <div className="toolInfo">
            <div className="toolTitle">PRACTICE TRADING</div>
            <div className="toolDesc">Simulation environment</div>
          </div>
        </button>
        <button 
          className={`toolBox ${activeTools.historyTool ? 'active' : ''}`} 
          onClick={() => handleToolClick('historyTool', 'launch-history-app')}
          disabled={activeTools.historyTool}
        >
          <div className="toolIconBox">
            <span className="toolIcon"></span>
          </div>
          <div className="toolInfo">
            <div className="toolTitle">HISTORICAL DATA</div>
            <div className="toolDesc">Market performance analysis</div>
          </div>
        </button>
        <button 
          className={`toolBox ${activeTools.brainTool ? 'active' : ''}`} 
          onClick={() => handleToolClick('brainTool', 'launch-brain-app')}
          disabled={activeTools.brainTool}
        >
          <div className="toolIconBox">
            <span className="toolIcon"></span>
          </div>
          <div className="toolInfo">
            <div className="toolTitle">TRADING LOG</div>
            <div className="toolDesc">Log Network</div>
          </div>
        </button>
        <button 
          className={`toolBox ${activeTools.analyticsTool ? 'active' : ''}`} 
          onClick={() => handleToolClick('analyticsTool', 'launch-analytics-app')}
          disabled={activeTools.analyticsTool}
        >
          <div className="toolIconBox">
            <span className="toolIcon"></span>
          </div>
          <div className="toolInfo">
            <div className="toolTitle">ADVANCED ANALYTICS</div>
            <div className="toolDesc">Market pattern analysis</div>
          </div>
        </button>
        <button 
          className={`toolBox ${activeTools.comingSoonTool ? 'active' : ''}`} 
          onClick={() => handleToolClick('comingSoonTool')}
          disabled={activeTools.comingSoonTool}
        >
          <div className="toolIconBox">
            <span className="toolIcon"></span>
          </div>
          <div className="toolInfo">
            <div className="toolTitle">COMING SOON</div>
            <div className="toolDesc">Future feature</div>
          </div>
        </button>
      </div>
    </div>
  );
};

export default Tools; 