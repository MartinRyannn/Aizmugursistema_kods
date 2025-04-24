import sys
import os
import uuid
import json
import re
import networkx as nx
import requests
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QTextEdit, QDialog, QTextBrowser,
                             QLabel, QSlider, QGroupBox, QGridLayout, QSpacerItem,
                             QSizePolicy)
from PyQt5.QtGui import QPainter, QColor, QPen, QLinearGradient, QRadialGradient, QTransform, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QPoint, pyqtSignal, QSize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import math

class TextSimilarityEngine:
    def __init__(self):
        self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                           'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                           'to', 'was', 'were', 'will', 'with'}
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        self.keyword_weights = {} 
        self.word_embeddings = {}  
        self.similarity_threshold = 0.45  
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        marked_tokens = []
        for token in tokens:
            if token in self.stop_words:
                marked_tokens.append('_stop_' + token)
            else:
                marked_tokens.append(token)
        return marked_tokens
    
    def update_word_embeddings(self, tokens):
        window_size = 4  
        
        for i, word in enumerate(tokens):
            if word.startswith('_stop_'):
                continue
                
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            context = tokens[start:i] + tokens[i+1:end]
            
            context = [w for w in context if not w.startswith('_stop_')]
            
            if word not in self.word_embeddings:
                self.word_embeddings[word] = {}
                
            for context_word in context:
                if context_word == word:
                    continue
                    
                if context_word not in self.word_embeddings[word]:
                    self.word_embeddings[word][context_word] = 0
                    
                self.word_embeddings[word][context_word] += 1
    
    def extract_keywords(self, text, update_global=True):
        processed = self.preprocess_text(text)
        
        if update_global:
            self.update_word_embeddings(processed)
        
        processed = [token.replace('_stop_', '') for token in processed 
                    if not token.startswith('_stop_')]
        
        term_freq = {}
        for token in processed:
            if len(token) > 2:
                term_freq[token] = term_freq.get(token, 0) + 1
        
        phrases = self._extract_phrases(text)
        for phrase in phrases:
            if phrase not in term_freq:
                term_freq[phrase] = 1
        
        keywords = {}
        total_tokens = len(processed) or 1
        for i, token in enumerate(processed):
            if token in term_freq:
                position_boost = 1.0
                if i < total_tokens * 0.25 or i > total_tokens * 0.75:
                    position_boost = 1.5
                
                score = (term_freq[token] / total_tokens) * position_boost
                keywords[token] = keywords.get(token, 0) + score
        
        for phrase in phrases:
            keywords[phrase] = keywords.get(phrase, 0) + 0.8  
            
        if update_global:
            for token, score in keywords.items():
                self.keyword_weights[token] = self.keyword_weights.get(token, 0) + score
        
        return keywords
    
    def _extract_phrases(self, text):
        text = text.lower()
        common_phrases = [
            "stuck to plan", "followed plan", "daily plan", 
            "trading plan", "fvg", "fair value gap", "support level",
            "resistance level", "price action", "market structure",
            "trend line", "break of structure", "bos", "change of character"
        ]
        
        phrases = []
        for phrase in common_phrases:
            if phrase in text:
                phrases.append(phrase)
                
        words = re.findall(r'\b\w+\b', text)
        for i in range(len(words) - 1):
            potential_phrase = f"{words[i]} {words[i+1]}"
            if len(potential_phrase) > 5 and potential_phrase not in common_phrases:
                phrases.append(potential_phrase)
                
        return phrases
    
    def compute_text_similarity(self, text1, text2):
        processed1 = self.preprocess_text(text1)
        processed2 = self.preprocess_text(text2)
        
        clean1 = [t.replace('_stop_', '') for t in processed1 if not t.startswith('_stop_')]
        clean2 = [t.replace('_stop_', '') for t in processed2 if not t.startswith('_stop_')]
        
        if not clean1 or not clean2:
            return 0
            
        unique_words1 = set(clean1)
        unique_words2 = set(clean2)
        jaccard_score = len(unique_words1.intersection(unique_words2)) / max(len(unique_words1.union(unique_words2)), 1)
        
        keywords1 = self.extract_keywords(text1, update_global=False)
        keywords2 = self.extract_keywords(text2, update_global=False)
        
        semantic_keyword_score = self._compute_semantic_keyword_similarity(keywords1, keywords2)
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([' '.join(clean1), ' '.join(clean2)])
            tfidf_score = cosine_similarity(tfidf_matrix)[0][1]
        except:
            tfidf_score = 0
        
        sequence_score = self._compute_word_sequence_similarity(clean1, clean2)
        
        phrases1 = self._extract_phrases(text1)
        phrases2 = self._extract_phrases(text2)
        phrase_overlap = len(set(phrases1).intersection(set(phrases2)))
        phrase_score = phrase_overlap / max(len(set(phrases1).union(set(phrases2))), 1)
        
        combined_score = (
            jaccard_score * 0.15 +
            semantic_keyword_score * 0.3 +
            tfidf_score * 0.2 +
            sequence_score * 0.15 +
            phrase_score * 0.2
        )
        
        if combined_score > 0.3:
            combined_score = min(combined_score * 1.5, 1.0)
            
        return combined_score
    
    def _compute_semantic_keyword_similarity(self, keywords1, keywords2):
        if not keywords1 or not keywords2:
            return 0
            
        all_keywords = set(keywords1.keys()).union(set(keywords2.keys()))
        direct_overlap = 0
        
        for keyword in all_keywords:
            if keyword in keywords1 and keyword in keywords2:
                direct_overlap += min(keywords1[keyword], keywords2[keyword])
        
        direct_score = direct_overlap / len(all_keywords) if all_keywords else 0
        
        semantic_score = 0
        match_count = 0
        
        for k1 in keywords1:
            best_match = 0
            
            for k2 in keywords2:
                if k1 == k2:
                    best_match = 1
                    break
                    
                similarity = self._word_semantic_similarity(k1, k2)
                best_match = max(best_match, similarity)
            
            semantic_score += best_match
            match_count += 1
        
        semantic_score = semantic_score / match_count if match_count > 0 else 0
        combined_score = direct_score * 0.6 + semantic_score * 0.4
        
        return combined_score
    
    def _word_semantic_similarity(self, word1, word2):
        if word1 == word2:
            return 1.0
            
        if word1 in word2 or word2 in word1:
            return 0.8
            
        if word1 in self.word_embeddings and word2 in self.word_embeddings[word1]:
            return min(self.word_embeddings[word1][word2] / 5, 0.7)  # Cap at 0.7
            
        if word2 in self.word_embeddings and word1 in self.word_embeddings[word2]:
            return min(self.word_embeddings[word2][word1] / 5, 0.7)  # Cap at 0.7
            
        shared_context = 0
        if word1 in self.word_embeddings and word2 in self.word_embeddings:
            context1 = set(self.word_embeddings[word1].keys())
            context2 = set(self.word_embeddings[word2].keys())
            overlap = context1.intersection(context2)
            if overlap:
                shared_context = len(overlap) / max(len(context1), len(context2), 1)
                return min(shared_context, 0.6)  # Cap at 0.6
                
        return self._string_similarity(word1, word2) * 0.3  
    
    def _string_similarity(self, s1, s2):
        if s1 == s2:
            return 1.0
            
        if s1 in s2 or s2 in s1:
            return 0.7
            
        longer = s1 if len(s1) >= len(s2) else s2
        shorter = s2 if len(s1) >= len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
            
        matches = sum(c1 == c2 for c1, c2 in zip(shorter, longer))
        return matches / len(longer)
        
    def _compute_word_sequence_similarity(self, seq1, seq2):
        n = len(seq1)
        m = len(seq2)
        
        if n == 0 or m == 0:
            return 0
            
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    similarity = self._string_similarity(seq1[i-1], seq2[j-1])
                    if similarity > 0.7:
                        dp[i][j] = dp[i-1][j-1] + similarity
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        sequence_similarity = 2 * dp[n][m] / (n + m)
        return sequence_similarity
    
    def get_top_keywords(self, n=10):
        sorted_keywords = sorted(self.keyword_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:n]

class NetworkNode:
    def __init__(self, id, analysis, trade_type, x=None, y=None):
        self.id = id
        self.analysis = analysis
        self.trade_type = trade_type
        
        spread_x = 600
        spread_y = 400
        center_x = 400
        center_y = 300
        
        self.x = x if x is not None else center_x + random.uniform(-spread_x/2, spread_x/2)
        self.y = y if y is not None else center_y + random.uniform(-spread_y/2, spread_y/2)
        
        self.base_size = 5
        self.size = self.base_size
        self.max_size = 20
        
        if trade_type == 'win':
            base_color = QColor(50, 205, 50)
            self.color = QColor(base_color.red(), base_color.green(), base_color.blue(), 180)
        else:
            base_color = QColor(220, 20, 60)
            self.color = QColor(base_color.red(), base_color.green(), base_color.blue(), 180)
        
        self.highlight_color = QColor(255, 215, 0, 200)
        self.connections = []
        self.similarity_count = 1
        self.idle_animation_phase = random.uniform(0, 2 * math.pi)
        
        self.keywords = {}
        self.content_similarity_score = 0
        self.connection_strength = {}
        
    def update_size_based_on_connections(self, keywords_weight=0.5, connection_weight=0.5):
        connection_factor = min(len(self.connections), 20) / 20 
        
        keyword_importance = 0
        if self.keywords:
            keyword_importance = sum(self.keywords.values()) / len(self.keywords)
        
        combined_factor = (
            connection_factor * connection_weight + 
            keyword_importance * keywords_weight
        )
        
        self.base_size = self.base_size + (self.max_size - self.base_size) * combined_factor
        self.size = self.base_size
        
    def add_connection(self, other_node, similarity_score):
        if other_node not in self.connections:
            self.connections.append(other_node)
            self.connection_strength[other_node.id] = similarity_score
        else:
            self.connection_strength[other_node.id] = max(
                self.connection_strength.get(other_node.id, 0),
                similarity_score
            )
    
    def calculate_centrality(self):
        return len(self.connections)
    
    def to_dict(self):
        return {
            'id': self.id,
            'analysis': self.analysis,
            'trade_type': self.trade_type,
            'x': self.x,
            'y': self.y,
            'base_size': self.base_size,
            'connections': [conn.id for conn in self.connections],
            'keywords': self.keywords,
            'connection_strength': self.connection_strength
        }

class TradingAnalysisNetwork:
    def __init__(self, save_file='trade_network.json'):
        self.graph = nx.Graph()
        self.nodes = {}
        self.similarity_engine = TextSimilarityEngine()
        self.similarity_threshold = 0.4
        self.save_file = save_file
        self.global_keywords = {} 
        self.keyword_boost_factors = {
            "fvg": 1.5,
            "plan": 1.5,
            "stuck": 1.4,
            "followed": 1.4,
            "daily": 1.3
        }
        self.load_network()

    def log_trade(self, analysis, trade_type='trade'):
        node_id = str(uuid.uuid4())
        node = NetworkNode(node_id, analysis, trade_type)
        
        extracted_keywords = self.similarity_engine.extract_keywords(analysis)
        
        node.keywords = {}
        for keyword, score in extracted_keywords.items():
            boost = 1.0
            for term, factor in self.keyword_boost_factors.items():
                if term in keyword:
                    boost = max(boost, factor)
            
            node.keywords[keyword] = score * boost
        
        for keyword, score in node.keywords.items():
            self.global_keywords[keyword] = self.global_keywords.get(keyword, 0) + score
        
        similar_nodes = self._find_similar_nodes(node)
        
        if similar_nodes:
            for similar_node, similarity in similar_nodes:
                node.add_connection(similar_node, similarity)
                similar_node.add_connection(node, similarity)
                similar_node.update_size_based_on_connections()
            
            node.update_size_based_on_connections()
        
        self.nodes[node_id] = node
        
        self._update_graph()
        
        self.save_network()
        return node_id

    def _find_similar_nodes(self, new_node):
        similar_nodes = []
        
        similarities = []
        for existing_node in self.nodes.values():
            type_similarity_factor = 1.0 if existing_node.trade_type == new_node.trade_type else 0.6
            
            base_similarity = self.similarity_engine.compute_text_similarity(
                existing_node.analysis, 
                new_node.analysis
            )
            
            adjusted_similarity = base_similarity * type_similarity_factor
            
            keyword_overlap = self._calculate_keyword_overlap(existing_node.keywords, new_node.keywords)
            
            enhanced_similarity = adjusted_similarity * (1.0 + keyword_overlap * 0.5)
            
            final_similarity = min(enhanced_similarity, 1.0)
            
            if final_similarity > self.similarity_threshold:
                similarities.append((existing_node, final_similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:min(len(similarities), 10)]
    
    def _calculate_keyword_overlap(self, keywords1, keywords2):
        if not keywords1 or not keywords2:
            return 0
            
        keys1 = set(keywords1.keys())
        keys2 = set(keywords2.keys())
        
        overlap = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        if not union:
            return 0
            
        weighted_overlap = 0
        for key in overlap:
            weight1 = keywords1.get(key, 0)
            weight2 = keywords2.get(key, 0)
            weighted_overlap += min(weight1, weight2)
            
        total_weight = sum(keywords1.values()) + sum(keywords2.values())
        if total_weight == 0:
            return 0
            
        return 2 * weighted_overlap / total_weight

    def _update_graph(self):
        self.graph = nx.Graph()
        
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, 
                               trade_type=node.trade_type, 
                               keywords=node.keywords,
                               analysis=node.analysis,
                               semantic_importance=sum(node.keywords.values()))
        
        for node_id, node in self.nodes.items():
            for conn in node.connections:
                strength = node.connection_strength.get(conn.id, 0.5)
                self.graph.add_edge(node_id, conn.id, weight=strength)
    
    def get_network_statistics(self):
        if len(self.nodes) < 2:
            return {
                "node_count": len(self.nodes),
                "edge_count": 0,
                "density": 0,
                "avg_clustering": 0,
                "top_keywords": [],
                "centrality": {},
                "communities": []
            }
        
        node_count = len(self.graph.nodes)
        edge_count = len(self.graph.edges)
        
        try:
            density = nx.density(self.graph)
            avg_clustering = nx.average_clustering(self.graph)
            
            degree_centrality = nx.degree_centrality(self.graph)
            
            betweenness_centrality = {}
            if node_count > 2:
                betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            communities = []
            if node_count > 3:
                try:
                    communities_iter = nx.community.girvan_newman(self.graph)
                    communities = next(communities_iter)
                    communities = [list(c) for c in communities]
                except:
                    communities = [list(c) for c in nx.connected_components(self.graph)]
            
            top_keywords_scored = sorted(self.global_keywords.items(), key=lambda x: x[1], reverse=True)
            top_keywords = top_keywords_scored[:15]
            
            keyword_patterns = self._analyze_keyword_patterns()
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": density,
                "avg_clustering": avg_clustering,
                "top_keywords": top_keywords,
                "degree_centrality": degree_centrality,
                "betweenness_centrality": betweenness_centrality,
                "communities": communities,
                "keyword_patterns": keyword_patterns
            }
        
        except Exception as e:
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": edge_count / (node_count * (node_count - 1) / 2) if node_count > 1 else 0,
                "top_keywords": sorted(self.global_keywords.items(), key=lambda x: x[1], reverse=True)[:10],
                "error": str(e)
            }
            
    def _analyze_keyword_patterns(self):
        keyword_co_occurrence = {}
        
        for node in self.nodes.values():
            node_keywords = list(node.keywords.keys())
            
            for i in range(len(node_keywords)):
                k1 = node_keywords[i]
                if k1 not in keyword_co_occurrence:
                    keyword_co_occurrence[k1] = {}
                    
                for j in range(i+1, len(node_keywords)):
                    k2 = node_keywords[j]
                    keyword_co_occurrence[k1][k2] = keyword_co_occurrence[k1].get(k2, 0) + 1
                    
                    if k2 not in keyword_co_occurrence:
                        keyword_co_occurrence[k2] = {}
                    keyword_co_occurrence[k2][k1] = keyword_co_occurrence[k2].get(k1, 0) + 1
        
        patterns = []
        for k1, co_words in keyword_co_occurrence.items():
            top_co_occurring = sorted(co_words.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_co_occurring:
                patterns.append({
                    'keyword': k1,
                    'co_occurring': top_co_occurring
                })
        
        patterns.sort(key=lambda x: self.global_keywords.get(x['keyword'], 0), reverse=True)
        
        return patterns[:10]

    def save_network(self):
        network_data = {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'global_keywords': self.global_keywords,
            'similarity_threshold': self.similarity_threshold
        }
        with open(self.save_file, 'w') as f:
            json.dump(network_data, f, indent=2)

    def load_network(self):
        if not os.path.exists(self.save_file):
            return

        try:
            with open(self.save_file, 'r') as f:
                content = f.read().strip()
                if not content:  # If file is empty
                    print("Network file is empty, starting with fresh network")
                    return
                    
                network_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error loading network file: {e}")
            print("Starting with fresh network")
            return
        except Exception as e:
            print(f"Unexpected error loading network: {e}")
            print("Starting with fresh network")
            return
        
        self.global_keywords = network_data.get('global_keywords', {})
        
        if 'similarity_threshold' in network_data:
            self.similarity_threshold = network_data['similarity_threshold']

        for node_data in network_data.get('nodes', []):
            node = NetworkNode(
                node_data['id'], 
                node_data['analysis'], 
                node_data['trade_type'], 
                node_data['x'], 
                node_data['y']
            )
            
            if 'base_size' in node_data:
                node.base_size = node_data['base_size']
                node.size = node.base_size
            
            if 'keywords' in node_data:
                node.keywords = node_data['keywords']
            
            if 'connection_strength' in node_data:
                node.connection_strength = node_data['connection_strength']
                
            self.nodes[node.id] = node

        for node_data in network_data.get('nodes', []):
            node = self.nodes[node_data['id']]
            for conn_id in node_data.get('connections', []):
                if conn_id in self.nodes:
                    conn_node = self.nodes[conn_id]
                    if conn_node not in node.connections:
                        node.connections.append(conn_node)
        
        self._update_graph()

class NetworkVisualizationWidget(QWidget):
    nodeClicked = pyqtSignal(object)
    
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.setMinimumSize(1000, 600)
        self.setStyleSheet("background-color: #121212;")
        
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_nodes)
        self.animation_timer.start(50)
        
        self.hovered_node = None
        self.selected_node = None
        
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_panning = False
        self.last_pan_pos = None
        
        self.animation_speed = 1.0
        self.animation_enabled = True
        self.animation_frame_count = 0
        self.animation_batch_size = 5

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            clicked_node = self.find_node_at_position(event.pos())
            if clicked_node:
                self.selected_node = clicked_node
                self.nodeClicked.emit(clicked_node)
                self.update()
            else:
                self.is_panning = True
                self.last_pan_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            dx = event.x() - self.last_pan_pos.x()
            dy = event.y() - self.last_pan_pos.y()
            self.offset_x += dx / self.zoom_level
            self.offset_y += dy / self.zoom_level
            self.last_pan_pos = event.pos()
            self.update()
        else:
            self.hovered_node = self.find_node_at_position(event.pos())
            self.update()

    def find_node_at_position(self, pos):
        for node in self.network.nodes.values():
            screen_x, screen_y = self.world_to_screen(node.x, node.y)
            distance = np.sqrt((pos.x() - screen_x)**2 + (pos.y() - screen_y)**2)
            if distance < node.size * self.zoom_level * 2:
                return node
        return None

    def world_to_screen(self, x, y):
        screen_x = (x + self.offset_x) * self.zoom_level
        screen_y = (y + self.offset_y) * self.zoom_level
        return screen_x, screen_y

    def screen_to_world(self, x, y):
        world_x = x / self.zoom_level - self.offset_x
        world_y = y / self.zoom_level - self.offset_y
        return world_x, world_y

    def set_zoom(self, zoom):
        old_zoom = self.zoom_level
        self.zoom_level = zoom
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        world_center_x, world_center_y = self.screen_to_world(center_x, center_y)
        
        self.offset_x = world_center_x - center_x / self.zoom_level
        self.offset_y = world_center_y - center_y / self.zoom_level
        
        self.update()

    def zoom_in(self):
        self.set_zoom(min(self.zoom_level * 1.2, 5.0))

    def zoom_out(self):
        self.set_zoom(max(self.zoom_level / 1.2, 0.2))

    def reset_view(self):
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update()

    def toggle_animation(self):
        self.animation_enabled = not self.animation_enabled
        return self.animation_enabled

    def set_animation_speed(self, speed):
        self.animation_speed = speed

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(18, 18, 18))
        gradient.setColorAt(1, QColor(30, 30, 30))
        painter.fillRect(self.rect(), gradient)
        
        if len(self.network.nodes) > 0:
            transform = QTransform()
            transform.translate(0, 0)
            painter.setTransform(transform)
            
            self.draw_all_connections(painter)
            
            self.draw_all_nodes(painter)

    def draw_all_connections(self, painter):
        painter.setPen(QPen(QColor(100, 100, 100, 30), 1, Qt.SolidLine))
        
        for node in self.network.nodes.values():
            node_x, node_y = self.world_to_screen(node.x, node.y)
            for connected_node in node.connections:
                conn_x, conn_y = self.world_to_screen(connected_node.x, connected_node.y)
                painter.drawLine(int(node_x), int(node_y), int(conn_x), int(conn_y))

    def draw_all_nodes(self, painter):
        for node in self.network.nodes.values():
            self.draw_node(painter, node)

    def draw_node(self, painter, node):
        glow_effect = 3 if node == self.hovered_node or node == self.selected_node else 1
        
        x, y = self.world_to_screen(node.x, node.y)
        center = QPointF(x, y)
        
        if (x < -100 or x > self.width() + 100 or 
            y < -100 or y > self.height() + 100):
            return
            
        node_size = node.size * self.zoom_level
        
        gradient = QRadialGradient(center, node_size * 3)
        
        base_color = node.highlight_color if node == self.selected_node else node.color
        gradient.setColorAt(0, QColor(base_color.red(), base_color.green(), base_color.blue(), 200 * glow_effect))
        gradient.setColorAt(1, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, node_size * glow_effect, node_size * glow_effect)
        
        painter.setBrush(base_color)
        painter.drawEllipse(center, node_size, node_size)

    def animate_nodes(self):
        if not self.animation_enabled or len(self.network.nodes) == 0:
            return
            
        # Batch processing for large networks
        self.animation_frame_count += 1
        if self.animation_frame_count % self.animation_batch_size != 0:
            return
            
        node_list = list(self.network.nodes.values())
        max_nodes_to_animate = min(len(node_list), 1000)  
        
        current_time = self.animation_frame_count / 20  
        
        for i in range(max_nodes_to_animate):
            node = node_list[i]
            animation_factor = 0.05 * self.animation_speed
            node.idle_animation_phase += animation_factor
            
            primary_wave = math.sin(node.idle_animation_phase) * 0.5
            secondary_wave = math.sin(node.idle_animation_phase * 2.5) * 0.2
            combined_effect = primary_wave + secondary_wave
            
            node.size = node.base_size + combined_effect
            
            if not self.selected_node or node != self.selected_node:
                x_offset = (math.sin(node.idle_animation_phase) * 
                            math.cos(node.idle_animation_phase * 0.5 + node.id.__hash__() % 10) * 
                            0.15 * self.animation_speed)
                
                y_offset = (math.cos(node.idle_animation_phase) * 
                            math.sin(node.idle_animation_phase * 0.5 + node.id.__hash__() % 10) * 
                            0.15 * self.animation_speed)
                
                node.x += x_offset
                node.y += y_offset
                
        self.update()


class ControlPanel(QWidget):
    def __init__(self, network_widget):
        super().__init__()
        self.network_widget = network_widget
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        nav_group = QGroupBox("Navigation")
        nav_layout = QGridLayout()
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.reset_view_btn = QPushButton("Reset View")
        
        nav_layout.addWidget(self.zoom_in_btn, 0, 0)
        nav_layout.addWidget(self.zoom_out_btn, 0, 1)
        nav_layout.addWidget(self.reset_view_btn, 1, 0, 1, 2)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        anim_group = QGroupBox("Animation")
        anim_layout = QVBoxLayout()
        
        self.toggle_anim_btn = QPushButton("Pause Animation")
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(150)
        speed_layout.addWidget(self.speed_slider)
        
        anim_layout.addWidget(self.toggle_anim_btn)
        anim_layout.addLayout(speed_layout)
        
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)
        
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("Nodes: 0\nConnections: 0")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        self.setLayout(layout)
        self.connectSignals()
        
    def connectSignals(self):
        self.zoom_in_btn.clicked.connect(self.network_widget.zoom_in)
        self.zoom_out_btn.clicked.connect(self.network_widget.zoom_out)
        self.reset_view_btn.clicked.connect(self.network_widget.reset_view)
        self.toggle_anim_btn.clicked.connect(self.toggleAnimation)
        self.speed_slider.valueChanged.connect(self.updateAnimationSpeed)
        
    def toggleAnimation(self):
        enabled = self.network_widget.toggle_animation()
        self.toggle_anim_btn.setText("Resume Animation" if not enabled else "Pause Animation")
        
    def updateAnimationSpeed(self, value):
        self.network_widget.set_animation_speed(value / 100.0)
        
    def updateStats(self, network):
        node_count = len(network.nodes)
        
        connection_count = 0
        for node in network.nodes.values():
            connection_count += len(node.connections)
        
        connection_count = connection_count // 2
        
        self.stats_label.setText(f"Nodes: {node_count}\nConnections: {connection_count}")

class NodeDetailsPanel(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        
        self.title_label = QLabel("Node Details")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.type_label = QLabel("Type: None")
        self.connections_label = QLabel("Connections: 0")
        
        self.analysis_browser = QTextBrowser()
        self.analysis_browser.setPlaceholderText("No node selected")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.type_label)
        layout.addWidget(self.connections_label)
        layout.addWidget(self.analysis_browser)
        
        self.setLayout(layout)
        
    def updateDetails(self, node):
        if node:
            self.title_label.setText(f"Node ID: {node.id[:8]}...")
            self.type_label.setText(f"Type: {node.trade_type.upper()}")
            self.connections_label.setText(f"Connections: {len(node.connections)}")
            self.analysis_browser.setText(node.analysis)
        else:
            self.title_label.setText("Node Details")
            self.type_label.setText("Type: None")
            self.connections_label.setText("Connections: 0")
            self.analysis_browser.clear()

class TradingAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Trading Analysis Network")
        self.setGeometry(100, 100, 1000, 500)
        
        self.network = TradingAnalysisNetwork()
        
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        left_panel = QVBoxLayout()
        
        input_group = QGroupBox("New Trade")
        input_layout = QVBoxLayout()
        
        self.analysis_input = QTextEdit()
        self.analysis_input.setPlaceholderText("Enter trade analysis...")
        
        trade_type_layout = QHBoxLayout()
        self.win_button = QPushButton("Win Trade")
        self.win_button.setStyleSheet("background-color: #32CD32; color: white;")
        self.loss_button = QPushButton("Loss Trade")
        self.loss_button.setStyleSheet("background-color: #DC143C; color: white;")
        
        trade_type_layout.addWidget(self.win_button)
        trade_type_layout.addWidget(self.loss_button)
        
        input_layout.addWidget(self.analysis_input)
        input_layout.addLayout(trade_type_layout)
        input_group.setLayout(input_layout)
        
        self.node_details = NodeDetailsPanel()
        
        left_panel.addWidget(input_group)
        left_panel.addWidget(self.node_details)
        
        middle_panel = QVBoxLayout()
        self.network_widget = NetworkVisualizationWidget(self.network)
        middle_panel.addWidget(self.network_widget)
        
        right_panel = QVBoxLayout()
        
        self.control_panel = ControlPanel(self.network_widget)
        right_panel.addWidget(self.control_panel)
        
        self.stats_panel = AdvancedStatisticsPanel(self.network)
        right_panel.addWidget(self.stats_panel)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(middle_panel, 5)
        main_layout.addLayout(right_panel, 2)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.win_button.clicked.connect(lambda: self.log_trade('win'))
        self.loss_button.clicked.connect(lambda: self.log_trade('loss'))
        self.network_widget.nodeClicked.connect(self.on_node_clicked)
        
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000)
        
        self.update_stats()
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E1E;
                color: #EEEEEE;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QTextEdit, QTextBrowser {
                background-color: #2D2D2D;
                border: 1px solid #555555;
                color: #EEEEEE;
            }
            QPushButton {
                background-color: #444444;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #666666;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4D4D4D;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #32CD32;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
    
    def log_trade(self, trade_type):
        analysis = self.analysis_input.toPlainText()
        if analysis:
            node_id = self.network.log_trade(analysis, trade_type)
            self.analysis_input.clear()
            
            self.update_stats()
            
            if node_id in self.network.nodes:
                new_node = self.network.nodes[node_id]
                self.network_widget.selected_node = new_node
                self.node_details.updateDetails(new_node)
                self.network_widget.update()
                
            self.statusBar().showMessage(f"New {trade_type} trade added to the network", 3000)
    
    def on_node_clicked(self, node):
        self.node_details.updateDetails(node)
        
        if hasattr(node, 'keywords') and node.keywords:
            keyword_info = "Top Keywords:\n"
            sorted_keywords = sorted(node.keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            for keyword, score in sorted_keywords:
                keyword_info += f"• {keyword}: {score:.3f}\n"
            self.node_details.analysis_browser.append("\n\n" + keyword_info)
            
    def update_stats(self):
        self.control_panel.updateStats(self.network)
        
        self.stats_panel.update_statistics()
        
        node_count = len(self.network.nodes)
        
        if node_count > 100:
            self.network_widget.animation_batch_size = 10
        elif node_count > 50:
            self.network_widget.animation_batch_size = 5
        else:
            self.network_widget.animation_batch_size = 1
    
    def analyze_network_trends(self):
        if len(self.network.nodes) < 5:
            return "Not enough data for trend analysis"
            
        stats = self.network.get_network_statistics()
        top_keywords = stats.get('top_keywords', [])
        
        win_nodes = [n for n in self.network.nodes.values() if n.trade_type == 'win']
        loss_nodes = [n for n in self.network.nodes.values() if n.trade_type == 'loss']
        
        win_keywords = {}
        for node in win_nodes:
            for kw, score in node.keywords.items():
                win_keywords[kw] = win_keywords.get(kw, 0) + score
                
        loss_keywords = {}
        for node in loss_nodes:
            for kw, score in node.keywords.items():
                loss_keywords[kw] = loss_keywords.get(kw, 0) + score
        
        distinctive_win = {}
        for kw, score in win_keywords.items():
            if kw in loss_keywords:
                if win_keywords[kw] / (len(win_nodes) or 1) > loss_keywords[kw] / (len(loss_nodes) or 1):
                    ratio = win_keywords[kw] / max(loss_keywords[kw], 0.1)
                    distinctive_win[kw] = ratio
            else:
                distinctive_win[kw] = score
                
        sorted_distinctive = sorted(distinctive_win.items(), key=lambda x: x[1], reverse=True)[:5]
        return sorted_distinctive
    
    def export_network_data(self, filename):
        export_data = {
            'nodes': [node.to_dict() for node in self.network.nodes.values()],
            'statistics': self.network.get_network_statistics(),
            'keywords': self.network.similarity_engine.get_top_keywords(20),
            'win_loss_ratio': len([n for n in self.network.nodes.values() if n.trade_type == 'win']) / 
                             max(len([n for n in self.network.nodes.values() if n.trade_type == 'loss']), 1)
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return True
    
    def closeEvent(self, event):
        self.network.save_network()
        event.accept()

class AdvancedStatisticsPanel(QWidget):
    def __init__(self, network):
        super().__init__()
        self.network = network
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        title_label = QLabel("Top Keywords")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        keywords_group = QGroupBox("Keywords")
        keywords_layout = QVBoxLayout()
        self.keywords_browser = QTextBrowser()
        self.keywords_browser.setMaximumHeight(300)
        keywords_layout.addWidget(self.keywords_browser)
        keywords_group.setLayout(keywords_layout)
        layout.addWidget(keywords_group)
        
        refresh_btn = QPushButton("Refresh Keywords")
        refresh_btn.clicked.connect(self.update_statistics)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        
        self.setLayout(layout)
        self.update_statistics()
        
    def update_statistics(self):
        stats = self.network.get_network_statistics()
        
        keywords_html = "<h3>Top Keywords</h3>"
        if stats.get('top_keywords'):
            keywords_html += "<table width='100%'>"
            keywords_html += "<tr><th>Keyword</th><th>Weight</th></tr>"
            for keyword, weight in stats['top_keywords']:
                keywords_html += f"<tr><td>{keyword}</td><td>{weight:.4f}</td></tr>"
            keywords_html += "</table>"
        else:
            keywords_html += "<p>No keywords found</p>"
            
        self.keywords_browser.setHtml(keywords_html)

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Advanced Trading Analysis Network")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #32CD32;")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Loading network data...")
        subtitle.setStyleSheet("font-size: 14px; color: #CCCCCC;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #121212; border: 1px solid #444444;")
        
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

def closeEvent(self, event):
    try:
        response = requests.post("http://localhost:3001/brain-closed")
        if response.status_code == 200:
            print("Successfully notified Flask app that the trading app is closing.")
        else:
            print(f"Failed to notify Flask app. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Error notifying Flask app: {e}")

    event.accept()

def main():
    app = QApplication(sys.argv)
    
    splash = SplashScreen()
    splash.show()
    
    app.processEvents()
    
    window = TradingAnalysisApp()
    window.show()
    
    splash.close()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()