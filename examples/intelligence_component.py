#!/usr/bin/env python3
"""
Intelligence Component - AI-OS SDK Example
A complete implementation of an intelligence component for pattern recognition and learning.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import json
import time
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from vertixia_sdk import AIIntelligenceComponent


class PatternRecognitionIntelligence(AIIntelligenceComponent):
    """
    Pattern Recognition Intelligence Component
    
    Analyzes data patterns, learns from observations, and provides insights.
    This component demonstrates:
    - Data pattern analysis
    - Machine learning integration
    - Anomaly detection
    - Predictive analytics
    - Knowledge extraction
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learned_patterns = {}
        self.anomaly_thresholds = {}
        self.prediction_models = {}
        self.knowledge_base = defaultdict(list)
    
    def _intelligence_execute(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                            analysis_type: str = "comprehensive", 
                            learning_enabled: bool = True,
                            **kwargs) -> Dict[str, Any]:
        """
        Execute intelligence analysis on input data
        
        Args:
            input_data: Data to analyze (single record or list of records)
            analysis_type: Type of analysis to perform
            learning_enabled: Whether to learn from the input data
            
        Returns:
            Dictionary containing analysis results and insights
        """
        start_time = time.time()
        
        try:
            # Normalize input data to list format
            if isinstance(input_data, dict):
                data_records = [input_data]
            else:
                data_records = input_data
            
            # Initialize results
            results = {
                "analysis_type": analysis_type,
                "records_analyzed": len(data_records),
                "processing_time": 0.0,
                "patterns_detected": [],
                "anomalies_detected": [],
                "insights": [],
                "predictions": {},
                "confidence_scores": {}
            }
            
            # Perform different types of analysis
            if analysis_type in ["pattern_detection", "comprehensive"]:
                pattern_results = self._detect_patterns(data_records)
                results["patterns_detected"] = pattern_results
            
            if analysis_type in ["anomaly_detection", "comprehensive"]:
                anomaly_results = self._detect_anomalies(data_records)
                results["anomalies_detected"] = anomaly_results
            
            if analysis_type in ["predictive", "comprehensive"]:
                prediction_results = self._generate_predictions(data_records)
                results["predictions"] = prediction_results
            
            if analysis_type in ["insight_generation", "comprehensive"]:
                insight_results = self._generate_insights(data_records)
                results["insights"] = insight_results
            
            # Learn from data if enabled
            if learning_enabled:
                learning_results = self._learn_from_data(data_records)
                results["learning_summary"] = learning_results
            
            # Calculate overall confidence scores
            results["confidence_scores"] = self._calculate_confidence_scores(results)
            
            # Update processing time
            results["processing_time"] = round(time.time() - start_time, 3)
            
            return {
                "success": True,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Intelligence analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _detect_patterns(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in the data"""
        self.logger.info("Detecting patterns in data")
        
        patterns = []
        
        # Temporal patterns
        temporal_patterns = self._detect_temporal_patterns(data_records)
        patterns.extend(temporal_patterns)
        
        # Frequency patterns
        frequency_patterns = self._detect_frequency_patterns(data_records)
        patterns.extend(frequency_patterns)
        
        # Correlation patterns
        correlation_patterns = self._detect_correlation_patterns(data_records)
        patterns.extend(correlation_patterns)
        
        # Sequence patterns
        sequence_patterns = self._detect_sequence_patterns(data_records)
        patterns.extend(sequence_patterns)
        
        return patterns
    
    def _detect_temporal_patterns(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect temporal patterns"""
        patterns = []
        
        # Group records by time periods
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for record in data_records:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hourly_counts[dt.hour] += 1
                    daily_counts[dt.strftime('%A')] += 1
                except:
                    continue
        
        # Detect peak hours
        if hourly_counts:
            peak_hour = max(hourly_counts, key=hourly_counts.get)
            peak_count = hourly_counts[peak_hour]
            avg_count = sum(hourly_counts.values()) / len(hourly_counts)
            
            if peak_count > avg_count * 1.5:  # 50% above average
                patterns.append({
                    "type": "temporal_peak",
                    "description": f"Peak activity at hour {peak_hour}",
                    "details": {
                        "peak_hour": peak_hour,
                        "peak_count": peak_count,
                        "average_count": round(avg_count, 2)
                    },
                    "confidence": min(0.9, peak_count / avg_count / 2)
                })
        
        # Detect day-of-week patterns
        if daily_counts:
            max_day = max(daily_counts, key=daily_counts.get)
            min_day = min(daily_counts, key=daily_counts.get)
            
            if daily_counts[max_day] > daily_counts[min_day] * 2:
                patterns.append({
                    "type": "day_of_week_pattern",
                    "description": f"Higher activity on {max_day}",
                    "details": {
                        "high_activity_day": max_day,
                        "low_activity_day": min_day,
                        "ratio": round(daily_counts[max_day] / daily_counts[min_day], 2)
                    },
                    "confidence": 0.8
                })
        
        return patterns
    
    def _detect_frequency_patterns(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect frequency patterns in categorical data"""
        patterns = []
        
        # Analyze categorical fields
        categorical_fields = self._identify_categorical_fields(data_records)
        
        for field in categorical_fields:
            values = [record.get(field) for record in data_records if record.get(field) is not None]
            if len(values) < 2:
                continue
            
            value_counts = Counter(values)
            total_count = len(values)
            
            # Find dominant values (>30% of data)
            dominant_values = {
                value: count for value, count in value_counts.items()
                if count / total_count > 0.3
            }
            
            if dominant_values:
                patterns.append({
                    "type": "frequency_pattern",
                    "description": f"Dominant values in field '{field}'",
                    "details": {
                        "field": field,
                        "dominant_values": {
                            value: {
                                "count": count,
                                "percentage": round(count / total_count * 100, 1)
                            }
                            for value, count in dominant_values.items()
                        }
                    },
                    "confidence": 0.7
                })
        
        return patterns
    
    def _detect_correlation_patterns(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect correlation patterns between numeric fields"""
        patterns = []
        
        # Extract numeric fields
        numeric_fields = self._identify_numeric_fields(data_records)
        
        if len(numeric_fields) < 2:
            return patterns
        
        # Calculate correlations between numeric fields
        for i, field1 in enumerate(numeric_fields):
            for field2 in numeric_fields[i+1:]:
                correlation = self._calculate_correlation(data_records, field1, field2)
                
                if abs(correlation) > 0.6:  # Strong correlation
                    patterns.append({
                        "type": "correlation_pattern",
                        "description": f"{'Strong positive' if correlation > 0 else 'Strong negative'} correlation between {field1} and {field2}",
                        "details": {
                            "field1": field1,
                            "field2": field2,
                            "correlation": round(correlation, 3),
                            "strength": "strong" if abs(correlation) > 0.8 else "moderate"
                        },
                        "confidence": abs(correlation)
                    })
        
        return patterns
    
    def _detect_sequence_patterns(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect sequence patterns in ordered data"""
        patterns = []
        
        # Sort records by timestamp if available
        sorted_records = self._sort_by_timestamp(data_records)
        if len(sorted_records) < 3:
            return patterns
        
        # Look for recurring sequences in categorical fields
        categorical_fields = self._identify_categorical_fields(sorted_records)
        
        for field in categorical_fields:
            sequences = self._find_recurring_sequences(sorted_records, field)
            
            for sequence, count in sequences.items():
                if count >= 2:  # Sequence appears at least twice
                    patterns.append({
                        "type": "sequence_pattern",
                        "description": f"Recurring sequence in field '{field}': {sequence}",
                        "details": {
                            "field": field,
                            "sequence": sequence.split(" -> "),
                            "occurrences": count,
                            "sequence_length": len(sequence.split(" -> "))
                        },
                        "confidence": min(0.9, count / 10)  # Higher confidence with more occurrences
                    })
        
        return patterns
    
    def _detect_anomalies(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        self.logger.info("Detecting anomalies in data")
        
        anomalies = []
        
        # Statistical anomalies in numeric fields
        numeric_anomalies = self._detect_statistical_anomalies(data_records)
        anomalies.extend(numeric_anomalies)
        
        # Behavioral anomalies
        behavioral_anomalies = self._detect_behavioral_anomalies(data_records)
        anomalies.extend(behavioral_anomalies)
        
        # Temporal anomalies
        temporal_anomalies = self._detect_temporal_anomalies(data_records)
        anomalies.extend(temporal_anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies using z-score method"""
        anomalies = []
        
        numeric_fields = self._identify_numeric_fields(data_records)
        
        for field in numeric_fields:
            values = [record.get(field) for record in data_records if record.get(field) is not None]
            if len(values) < 3:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                continue
            
            # Find outliers using z-score > 2.5
            for i, record in enumerate(data_records):
                value = record.get(field)
                if value is not None:
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > 2.5:
                        anomalies.append({
                            "type": "statistical_anomaly",
                            "description": f"Outlier value in field '{field}'",
                            "details": {
                                "record_index": i,
                                "field": field,
                                "value": value,
                                "z_score": round(z_score, 2),
                                "mean": round(mean_val, 2),
                                "std_dev": round(std_val, 2)
                            },
                            "severity": "high" if z_score > 3.5 else "medium",
                            "confidence": min(0.9, z_score / 4)
                        })
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies based on learned patterns"""
        anomalies = []
        
        # Compare against learned patterns
        for pattern_type, learned_patterns in self.learned_patterns.items():
            for record in data_records:
                if self._is_behavioral_anomaly(record, learned_patterns):
                    anomalies.append({
                        "type": "behavioral_anomaly",
                        "description": f"Behavior deviates from learned pattern '{pattern_type}'",
                        "details": {
                            "pattern_type": pattern_type,
                            "record": record,
                            "expected_pattern": learned_patterns
                        },
                        "severity": "medium",
                        "confidence": 0.7
                    })
        
        return anomalies
    
    def _generate_predictions(self, data_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate predictions based on historical patterns"""
        self.logger.info("Generating predictions")
        
        predictions = {}
        
        # Trend predictions
        trend_predictions = self._predict_trends(data_records)
        predictions["trends"] = trend_predictions
        
        # Next value predictions
        next_value_predictions = self._predict_next_values(data_records)
        predictions["next_values"] = next_value_predictions
        
        # Probability predictions
        probability_predictions = self._predict_probabilities(data_records)
        predictions["probabilities"] = probability_predictions
        
        return predictions
    
    def _generate_insights(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable insights from data"""
        self.logger.info("Generating insights")
        
        insights = []
        
        # Performance insights
        performance_insights = self._generate_performance_insights(data_records)
        insights.extend(performance_insights)
        
        # Optimization insights
        optimization_insights = self._generate_optimization_insights(data_records)
        insights.extend(optimization_insights)
        
        # Risk insights
        risk_insights = self._generate_risk_insights(data_records)
        insights.extend(risk_insights)
        
        return insights
    
    def _learn_from_data(self, data_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn patterns from new data"""
        self.logger.info("Learning from data")
        
        learning_summary = {
            "patterns_learned": 0,
            "models_updated": 0,
            "knowledge_items_added": 0
        }
        
        # Update learned patterns
        new_patterns = self._extract_learnable_patterns(data_records)
        for pattern_type, patterns in new_patterns.items():
            if pattern_type not in self.learned_patterns:
                self.learned_patterns[pattern_type] = []
            self.learned_patterns[pattern_type].extend(patterns)
            learning_summary["patterns_learned"] += len(patterns)
        
        # Update prediction models
        updated_models = self._update_prediction_models(data_records)
        learning_summary["models_updated"] = len(updated_models)
        
        # Add to knowledge base
        knowledge_items = self._extract_knowledge(data_records)
        for category, items in knowledge_items.items():
            self.knowledge_base[category].extend(items)
            learning_summary["knowledge_items_added"] += len(items)
        
        return learning_summary
    
    # Helper methods
    def _identify_categorical_fields(self, data_records: List[Dict[str, Any]]) -> List[str]:
        """Identify categorical fields in the data"""
        field_types = defaultdict(set)
        
        for record in data_records:
            for field, value in record.items():
                if isinstance(value, (str, bool)):
                    field_types[field].add(type(value).__name__)
                elif isinstance(value, (int, float)) and len(str(value)) < 3:
                    field_types[field].add("categorical_numeric")
        
        return [field for field, types in field_types.items() 
                if "str" in types or "bool" in types or "categorical_numeric" in types]
    
    def _identify_numeric_fields(self, data_records: List[Dict[str, Any]]) -> List[str]:
        """Identify numeric fields in the data"""
        numeric_fields = []
        
        for record in data_records:
            for field, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if field not in numeric_fields:
                        numeric_fields.append(field)
        
        return numeric_fields
    
    def _calculate_correlation(self, data_records: List[Dict[str, Any]], field1: str, field2: str) -> float:
        """Calculate correlation between two numeric fields"""
        values1 = []
        values2 = []
        
        for record in data_records:
            val1 = record.get(field1)
            val2 = record.get(field2)
            if val1 is not None and val2 is not None:
                values1.append(val1)
                values2.append(val2)
        
        if len(values1) < 2:
            return 0.0
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    def _sort_by_timestamp(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort records by timestamp"""
        timestamped_records = []
        
        for record in data_records:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamped_records.append((dt, record))
                except:
                    continue
        
        timestamped_records.sort(key=lambda x: x[0])
        return [record for _, record in timestamped_records]
    
    def _calculate_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence scores"""
        scores = {}
        
        # Pattern detection confidence
        if results["patterns_detected"]:
            avg_pattern_confidence = np.mean([p.get("confidence", 0.5) for p in results["patterns_detected"]])
            scores["pattern_detection"] = round(avg_pattern_confidence, 2)
        
        # Anomaly detection confidence
        if results["anomalies_detected"]:
            avg_anomaly_confidence = np.mean([a.get("confidence", 0.5) for a in results["anomalies_detected"]])
            scores["anomaly_detection"] = round(avg_anomaly_confidence, 2)
        
        # Overall confidence
        if scores:
            scores["overall"] = round(np.mean(list(scores.values())), 2)
        else:
            scores["overall"] = 0.5
        
        return scores


# Example usage and testing
if __name__ == "__main__":
    # Create intelligence component instance
    intelligence = PatternRecognitionIntelligence()
    
    # Example data for analysis
    sample_data = [
        {
            "timestamp": "2024-01-15T09:30:00Z",
            "user_id": "user_123",
            "action": "login",
            "duration": 45,
            "location": "office",
            "device": "laptop"
        },
        {
            "timestamp": "2024-01-15T09:35:00Z", 
            "user_id": "user_123",
            "action": "file_access",
            "duration": 120,
            "location": "office",
            "device": "laptop"
        },
        {
            "timestamp": "2024-01-15T14:15:00Z",
            "user_id": "user_456",
            "action": "login",
            "duration": 30,
            "location": "home",
            "device": "mobile"
        },
        {
            "timestamp": "2024-01-15T14:20:00Z",
            "user_id": "user_456",
            "action": "file_access",
            "duration": 300,  # Unusual duration
            "location": "home",
            "device": "mobile"
        }
    ]
    
    # Test comprehensive analysis
    print("=== Comprehensive Intelligence Analysis ===")
    result = intelligence.execute(
        input_data=sample_data,
        analysis_type="comprehensive",
        learning_enabled=True
    )
    
    if result["success"]:
        results = result["results"]
        print(f"Records analyzed: {results['records_analyzed']}")
        print(f"Processing time: {results['processing_time']}s")
        print(f"Patterns detected: {len(results['patterns_detected'])}")
        print(f"Anomalies detected: {len(results['anomalies_detected'])}")
        print(f"Insights generated: {len(results['insights'])}")
        
        # Print patterns
        for pattern in results["patterns_detected"]:
            print(f"\nPattern: {pattern['description']}")
            print(f"Type: {pattern['type']}")
            print(f"Confidence: {pattern['confidence']}")
        
        # Print anomalies
        for anomaly in results["anomalies_detected"]:
            print(f"\nAnomaly: {anomaly['description']}")
            print(f"Severity: {anomaly['severity']}")
            print(f"Confidence: {anomaly['confidence']}")
    
    # Test pattern-specific analysis
    print("\n=== Pattern Detection Only ===")
    pattern_result = intelligence.execute(
        input_data=sample_data,
        analysis_type="pattern_detection",
        learning_enabled=False
    )
    
    if pattern_result["success"]:
        patterns = pattern_result["results"]["patterns_detected"]
        print(f"Patterns found: {len(patterns)}")
        for pattern in patterns:
            print(f"- {pattern['description']} (confidence: {pattern['confidence']})")