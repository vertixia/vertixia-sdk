#!/usr/bin/env python3
"""
Automation Component - AI-OS SDK Example
A complete implementation of an automation component for task scheduling and execution.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import time
from datetime import datetime, timedelta
from vertixia_sdk import AIAutomationComponent


class TaskSchedulerAutomation(AIAutomationComponent):
    """
    Task Scheduler Automation Component
    
    Automates task scheduling, execution, and monitoring.
    This component demonstrates:
    - Event-driven automation
    - Schedule-based triggers
    - Task orchestration
    - Monitoring and alerting
    - Error handling and recovery
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheduled_tasks = {}
        self.task_history = []
        self.active_monitors = {}
    
    def _automation_execute(self, automation_config: Dict[str, Any], 
                          trigger_event: Optional[Dict[str, Any]] = None, 
                          **kwargs) -> Dict[str, Any]:
        """
        Execute automation based on configuration and triggers
        
        Args:
            automation_config: Automation configuration
            trigger_event: Event that triggered the automation
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        
        try:
            automation_type = automation_config.get("type", "scheduled")
            
            if automation_type == "scheduled":
                return self._handle_scheduled_automation(automation_config, trigger_event)
            elif automation_type == "event_driven":
                return self._handle_event_driven_automation(automation_config, trigger_event)
            elif automation_type == "monitoring":
                return self._handle_monitoring_automation(automation_config, trigger_event)
            elif automation_type == "workflow":
                return self._handle_workflow_automation(automation_config, trigger_event)
            else:
                raise ValueError(f"Unknown automation type: {automation_type}")
        
        except Exception as e:
            self.logger.error(f"Automation execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _handle_scheduled_automation(self, config: Dict[str, Any], 
                                   trigger_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle scheduled automation tasks"""
        self.logger.info("Executing scheduled automation")
        
        schedule = config.get("schedule", {})
        tasks = config.get("tasks", [])
        
        results = {
            "automation_type": "scheduled",
            "executed_tasks": [],
            "failed_tasks": [],
            "next_execution": None
        }
        
        # Execute scheduled tasks
        for task in tasks:
            try:
                task_result = self._execute_task(task)
                results["executed_tasks"].append({
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "status": "completed",
                    "result": task_result,
                    "execution_time": task_result.get("execution_time", 0)
                })
            except Exception as e:
                self.logger.error(f"Task execution failed: {str(e)}")
                results["failed_tasks"].append({
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "error": str(e)
                })
        
        # Calculate next execution time
        if schedule.get("recurring", False):
            interval = schedule.get("interval_minutes", 60)
            results["next_execution"] = datetime.now() + timedelta(minutes=interval)
        
        return {
            "success": True,
            "results": results
        }
    
    def _handle_event_driven_automation(self, config: Dict[str, Any], 
                                      trigger_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle event-driven automation"""
        self.logger.info("Executing event-driven automation")
        
        if not trigger_event:
            return {
                "success": False,
                "error": "No trigger event provided for event-driven automation"
            }
        
        event_type = trigger_event.get("type")
        event_data = trigger_event.get("data", {})
        
        # Find matching automation rules
        rules = config.get("rules", [])
        matching_rules = []
        
        for rule in rules:
            if self._event_matches_rule(trigger_event, rule):
                matching_rules.append(rule)
        
        if not matching_rules:
            return {
                "success": True,
                "message": "No matching rules found for event",
                "event_type": event_type
            }
        
        # Execute actions for matching rules
        executed_actions = []
        failed_actions = []
        
        for rule in matching_rules:
            actions = rule.get("actions", [])
            
            for action in actions:
                try:
                    action_result = self._execute_action(action, event_data)
                    executed_actions.append({
                        "rule_id": rule.get("id"),
                        "action_id": action.get("id"),
                        "action_type": action.get("type"),
                        "result": action_result
                    })
                except Exception as e:
                    self.logger.error(f"Action execution failed: {str(e)}")
                    failed_actions.append({
                        "rule_id": rule.get("id"),
                        "action_id": action.get("id"),
                        "error": str(e)
                    })
        
        return {
            "success": True,
            "results": {
                "automation_type": "event_driven",
                "trigger_event": trigger_event,
                "matching_rules": len(matching_rules),
                "executed_actions": executed_actions,
                "failed_actions": failed_actions
            }
        }
    
    def _handle_monitoring_automation(self, config: Dict[str, Any], 
                                    trigger_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle monitoring automation"""
        self.logger.info("Executing monitoring automation")
        
        monitors = config.get("monitors", [])
        monitoring_results = []
        
        for monitor in monitors:
            try:
                result = self._execute_monitor(monitor)
                monitoring_results.append({
                    "monitor_id": monitor.get("id"),
                    "monitor_name": monitor.get("name"),
                    "status": result.get("status"),
                    "metrics": result.get("metrics", {}),
                    "alerts": result.get("alerts", [])
                })
                
                # Handle alerts
                if result.get("alerts"):
                    self._handle_alerts(result["alerts"], monitor)
                    
            except Exception as e:
                self.logger.error(f"Monitor execution failed: {str(e)}")
                monitoring_results.append({
                    "monitor_id": monitor.get("id"),
                    "monitor_name": monitor.get("name"),
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": {
                "automation_type": "monitoring",
                "monitors_executed": len(monitors),
                "monitoring_results": monitoring_results
            }
        }
    
    def _handle_workflow_automation(self, config: Dict[str, Any], 
                                  trigger_event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle workflow automation"""
        self.logger.info("Executing workflow automation")
        
        workflow_steps = config.get("workflow_steps", [])
        execution_context = trigger_event.get("data", {}) if trigger_event else {}
        
        step_results = []
        current_step = 0
        
        for step in workflow_steps:
            try:
                step_result = self._execute_workflow_step(step, execution_context)
                
                step_results.append({
                    "step_number": current_step + 1,
                    "step_id": step.get("id"),
                    "step_name": step.get("name"),
                    "status": "completed",
                    "result": step_result,
                    "execution_time": step_result.get("execution_time", 0)
                })
                
                # Update execution context with step results
                if step_result.get("output"):
                    execution_context.update(step_result["output"])
                
                current_step += 1
                
                # Check for early termination conditions
                if step.get("terminate_on_success") and step_result.get("success"):
                    break
                
            except Exception as e:
                self.logger.error(f"Workflow step failed: {str(e)}")
                step_results.append({
                    "step_number": current_step + 1,
                    "step_id": step.get("id"),
                    "step_name": step.get("name"),
                    "status": "failed",
                    "error": str(e)
                })
                
                # Check if workflow should continue on error
                if not step.get("continue_on_error", False):
                    break
                
                current_step += 1
        
        workflow_success = all(
            result["status"] == "completed" 
            for result in step_results
        )
        
        return {
            "success": workflow_success,
            "results": {
                "automation_type": "workflow",
                "steps_executed": len(step_results),
                "workflow_success": workflow_success,
                "step_results": step_results,
                "final_context": execution_context
            }
        }
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task"""
        task_type = task.get("type")
        task_params = task.get("parameters", {})
        
        start_time = time.time()
        
        if task_type == "email_notification":
            return self._send_email_notification(task_params)
        elif task_type == "data_backup":
            return self._perform_data_backup(task_params)
        elif task_type == "system_cleanup":
            return self._perform_system_cleanup(task_params)
        elif task_type == "report_generation":
            return self._generate_report(task_params)
        elif task_type == "api_call":
            return self._make_api_call(task_params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _execute_action(self, action: Dict[str, Any], event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an automation action"""
        action_type = action.get("type")
        action_params = action.get("parameters", {})
        
        # Merge event data with action parameters
        merged_params = {**action_params, **event_data}
        
        if action_type == "send_alert":
            return self._send_alert(merged_params)
        elif action_type == "trigger_workflow":
            return self._trigger_workflow(merged_params)
        elif action_type == "update_database":
            return self._update_database(merged_params)
        elif action_type == "create_ticket":
            return self._create_support_ticket(merged_params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _execute_monitor(self, monitor: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a monitoring check"""
        monitor_type = monitor.get("type")
        monitor_params = monitor.get("parameters", {})
        
        if monitor_type == "system_health":
            return self._check_system_health(monitor_params)
        elif monitor_type == "application_performance":
            return self._check_application_performance(monitor_params)
        elif monitor_type == "data_quality":
            return self._check_data_quality(monitor_params)
        elif monitor_type == "security_scan":
            return self._perform_security_scan(monitor_params)
        else:
            raise ValueError(f"Unknown monitor type: {monitor_type}")
    
    def _execute_workflow_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step"""
        step_type = step.get("type")
        step_params = step.get("parameters", {})
        
        # Substitute context variables in parameters
        resolved_params = self._resolve_context_variables(step_params, context)
        
        start_time = time.time()
        
        if step_type == "condition_check":
            return self._evaluate_condition(resolved_params, context)
        elif step_type == "data_transformation":
            return self._transform_data(resolved_params, context)
        elif step_type == "external_service_call":
            return self._call_external_service(resolved_params)
        elif step_type == "wait":
            return self._wait_step(resolved_params)
        else:
            # Delegate to task execution for common task types
            return self._execute_task({"type": step_type, "parameters": resolved_params})
    
    def _event_matches_rule(self, event: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if an event matches an automation rule"""
        conditions = rule.get("conditions", [])
        
        for condition in conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            event_value = event.get("data", {}).get(field)
            
            if not self._evaluate_condition_operator(event_value, operator, value):
                return False
        
        return True
    
    def _evaluate_condition_operator(self, event_value: Any, operator: str, expected_value: Any) -> bool:
        """Evaluate a condition operator"""
        if operator == "equals":
            return event_value == expected_value
        elif operator == "not_equals":
            return event_value != expected_value
        elif operator == "greater_than":
            return event_value > expected_value
        elif operator == "less_than":
            return event_value < expected_value
        elif operator == "contains":
            return expected_value in str(event_value)
        elif operator == "starts_with":
            return str(event_value).startswith(str(expected_value))
        else:
            return False
    
    def _resolve_context_variables(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve context variables in parameters"""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable name
                var_name = value[2:-1]
                resolved[key] = context.get(var_name, value)
            else:
                resolved[key] = value
        
        return resolved
    
    # Simplified implementations of various task types
    def _send_email_notification(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification (simplified)"""
        return {
            "success": True,
            "message": f"Email sent to {params.get('recipient')}",
            "execution_time": 0.1
        }
    
    def _perform_data_backup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data backup (simplified)"""
        return {
            "success": True,
            "backup_size": "1.2GB",
            "backup_location": params.get("destination", "/backup"),
            "execution_time": 5.0
        }
    
    def _check_system_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check system health (simplified)"""
        # Simulate health check
        cpu_usage = 45.2
        memory_usage = 67.8
        disk_usage = 23.1
        
        alerts = []
        if cpu_usage > 80:
            alerts.append("High CPU usage detected")
        if memory_usage > 90:
            alerts.append("High memory usage detected")
        
        return {
            "status": "healthy" if not alerts else "warning",
            "metrics": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage
            },
            "alerts": alerts
        }
    
    def _handle_alerts(self, alerts: List[str], monitor: Dict[str, Any]) -> None:
        """Handle monitoring alerts"""
        for alert in alerts:
            self.logger.warning(f"Alert from {monitor.get('name')}: {alert}")
            
            # In a real implementation, this would send notifications
            # via email, Slack, PagerDuty, etc.


# Example usage and testing
if __name__ == "__main__":
    # Create automation component instance
    automation = TaskSchedulerAutomation()
    
    # Example 1: Scheduled automation
    print("=== Scheduled Automation ===")
    scheduled_config = {
        "type": "scheduled",
        "schedule": {
            "recurring": True,
            "interval_minutes": 60
        },
        "tasks": [
            {
                "id": "backup_task",
                "name": "Daily Backup",
                "type": "data_backup",
                "parameters": {
                    "source": "/data",
                    "destination": "/backup"
                }
            },
            {
                "id": "cleanup_task", 
                "name": "System Cleanup",
                "type": "system_cleanup",
                "parameters": {
                    "temp_files": True,
                    "log_retention_days": 30
                }
            }
        ]
    }
    
    result = automation.execute(automation_config=scheduled_config)
    print(f"Scheduled automation success: {result['success']}")
    if result['success']:
        results = result['results']
        print(f"Executed tasks: {len(results['executed_tasks'])}")
        print(f"Failed tasks: {len(results['failed_tasks'])}")
    
    # Example 2: Event-driven automation
    print("\n=== Event-Driven Automation ===")
    event_config = {
        "type": "event_driven",
        "rules": [
            {
                "id": "high_cpu_rule",
                "conditions": [
                    {
                        "field": "cpu_usage",
                        "operator": "greater_than",
                        "value": 80
                    }
                ],
                "actions": [
                    {
                        "id": "alert_action",
                        "type": "send_alert",
                        "parameters": {
                            "severity": "warning",
                            "message": "High CPU usage detected"
                        }
                    }
                ]
            }
        ]
    }
    
    trigger_event = {
        "type": "system_metric",
        "data": {
            "cpu_usage": 85.5,
            "memory_usage": 45.2,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    result = automation.execute(
        automation_config=event_config,
        trigger_event=trigger_event
    )
    print(f"Event-driven automation success: {result['success']}")
    if result['success']:
        results = result['results']
        print(f"Matching rules: {results['matching_rules']}")
        print(f"Executed actions: {len(results['executed_actions'])}")
    
    # Example 3: Monitoring automation
    print("\n=== Monitoring Automation ===")
    monitoring_config = {
        "type": "monitoring",
        "monitors": [
            {
                "id": "health_monitor",
                "name": "System Health Check",
                "type": "system_health",
                "parameters": {
                    "check_cpu": True,
                    "check_memory": True,
                    "check_disk": True
                }
            }
        ]
    }
    
    result = automation.execute(automation_config=monitoring_config)
    print(f"Monitoring automation success: {result['success']}")
    if result['success']:
        results = result['results']
        for monitor_result in results['monitoring_results']:
            print(f"Monitor: {monitor_result['monitor_name']}")
            print(f"Status: {monitor_result['status']}")
            if monitor_result.get('metrics'):
                for metric, value in monitor_result['metrics'].items():
                    print(f"  {metric}: {value}")