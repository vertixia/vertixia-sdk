#!/usr/bin/env python3
"""
Email Processing Agent - AI-OS SDK Example
A complete implementation of an autonomous email processing agent.
"""

from typing import Dict, Any, List, Union, Optional
import json
import time
from datetime import datetime, timedelta
from vertixia_sdk import AIAgentComponent


class EmailProcessingAgent(AIAgentComponent):
    """
    Email Processing Agent
    
    An autonomous agent that can read, process, categorize, and respond to emails.
    This agent demonstrates:
    - Task planning and decomposition
    - Service integration (Gmail API)
    - Context management
    - Decision making
    - Automated responses
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.email_categories = {
            "urgent": ["urgent", "asap", "emergency", "critical", "immediate"],
            "meeting": ["meeting", "call", "appointment", "schedule", "calendar"],
            "project": ["project", "task", "deadline", "deliverable", "milestone"],
            "support": ["help", "issue", "problem", "bug", "error", "support"],
            "newsletter": ["newsletter", "unsubscribe", "marketing", "promotion"],
            "personal": ["personal", "family", "friend", "social"]
        }
    
    def _plan_task(self, task: Union[str, Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Plan email processing task
        
        Args:
            task: Task description or structured task data
            
        Returns:
            List of planned steps
        """
        if isinstance(task, str):
            # Parse natural language task
            task_data = self._parse_task_description(task)
        else:
            task_data = task
        
        # Create execution plan based on task type
        plan = []
        
        # Step 1: Always start by checking for new emails
        plan.append({
            "step": 1,
            "action": "fetch_emails",
            "description": "Fetch recent emails from inbox",
            "parameters": {
                "max_results": task_data.get("max_emails", 10),
                "query": task_data.get("email_query", "is:unread"),
                "time_range": task_data.get("time_range", "1d")
            }
        })
        
        # Step 2: Process and categorize emails
        plan.append({
            "step": 2,
            "action": "categorize_emails",
            "description": "Categorize emails by type and priority",
            "parameters": {
                "categories": list(self.email_categories.keys())
            }
        })
        
        # Step 3: Handle urgent emails first
        if task_data.get("handle_urgent", True):
            plan.append({
                "step": 3,
                "action": "process_urgent_emails",
                "description": "Prioritize and handle urgent emails",
                "parameters": {
                    "auto_respond": task_data.get("auto_respond", False),
                    "escalate_threshold": task_data.get("escalate_threshold", 2)
                }
            })
        
        # Step 4: Process other emails based on category
        plan.append({
            "step": 4,
            "action": "process_categorized_emails",
            "description": "Process emails by category",
            "parameters": {
                "process_meetings": task_data.get("process_meetings", True),
                "process_projects": task_data.get("process_projects", True),
                "auto_archive_newsletters": task_data.get("auto_archive_newsletters", False)
            }
        })
        
        # Step 5: Generate summary report
        plan.append({
            "step": 5,
            "action": "generate_summary",
            "description": "Generate email processing summary",
            "parameters": {
                "include_statistics": True,
                "include_action_items": True
            }
        })
        
        self.logger.info(f"Generated execution plan with {len(plan)} steps")
        return plan
    
    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step in the plan
        
        Args:
            step: Step to execute
            context: Current execution context
            
        Returns:
            Step execution result
        """
        action = step["action"]
        parameters = step.get("parameters", {})
        
        try:
            if action == "fetch_emails":
                return self._fetch_emails(parameters, context)
            elif action == "categorize_emails":
                return self._categorize_emails(parameters, context)
            elif action == "process_urgent_emails":
                return self._process_urgent_emails(parameters, context)
            elif action == "process_categorized_emails":
                return self._process_categorized_emails(parameters, context)
            elif action == "generate_summary":
                return self._generate_summary(parameters, context)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "step": step["step"],
                "action": action
            }
    
    def _parse_task_description(self, task: str) -> Dict[str, Any]:
        """Parse natural language task description"""
        task_lower = task.lower()
        
        # Extract parameters from natural language
        task_data = {
            "max_emails": 20,
            "email_query": "is:unread",
            "time_range": "1d",
            "handle_urgent": True,
            "auto_respond": False,
            "process_meetings": True,
            "process_projects": True,
            "auto_archive_newsletters": False
        }
        
        # Parse specific requirements
        if "urgent" in task_lower:
            task_data["handle_urgent"] = True
            task_data["escalate_threshold"] = 1
        
        if "respond" in task_lower or "reply" in task_lower:
            task_data["auto_respond"] = True
        
        if "meeting" in task_lower or "calendar" in task_lower:
            task_data["process_meetings"] = True
        
        if "archive" in task_lower and "newsletter" in task_lower:
            task_data["auto_archive_newsletters"] = True
        
        # Parse time range
        if "today" in task_lower:
            task_data["time_range"] = "1d"
        elif "week" in task_lower:
            task_data["time_range"] = "7d"
        elif "hour" in task_lower:
            task_data["time_range"] = "1h"
        
        return task_data
    
    def _fetch_emails(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch emails from inbox"""
        self.logger.info("Fetching emails from inbox")
        
        # In a real implementation, this would call Gmail API
        # For this example, we'll simulate email data
        simulated_emails = [
            {
                "id": "email_1",
                "subject": "URGENT: Server down - immediate action required",
                "sender": "alerts@company.com",
                "received_time": datetime.now() - timedelta(minutes=15),
                "body": "The production server is experiencing critical issues...",
                "labels": ["INBOX", "UNREAD"]
            },
            {
                "id": "email_2",
                "subject": "Meeting invitation: Project Review",
                "sender": "manager@company.com",
                "received_time": datetime.now() - timedelta(hours=2),
                "body": "Please join us for the weekly project review meeting...",
                "labels": ["INBOX", "UNREAD"]
            },
            {
                "id": "email_3",
                "subject": "Weekly Newsletter - AI Trends",
                "sender": "newsletter@aitrends.com",
                "received_time": datetime.now() - timedelta(hours=6),
                "body": "This week's top AI and machine learning news...",
                "labels": ["INBOX", "UNREAD"]
            },
            {
                "id": "email_4",
                "subject": "Project deadline reminder",
                "sender": "project@company.com",
                "received_time": datetime.now() - timedelta(hours=12),
                "body": "Reminder: The AI-OS project milestone is due...",
                "labels": ["INBOX", "UNREAD"]
            }
        ]
        
        # Apply filters based on parameters
        max_results = parameters.get("max_results", 10)
        emails = simulated_emails[:max_results]
        
        # Store emails in context
        context["emails"] = emails
        context["email_count"] = len(emails)
        
        return {
            "success": True,
            "emails_fetched": len(emails),
            "emails": emails
        }
    
    def _categorize_emails(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize emails by type and priority"""
        self.logger.info("Categorizing emails")
        
        emails = context.get("emails", [])
        categorized = {category: [] for category in self.email_categories.keys()}
        categorized["other"] = []
        
        for email in emails:
            subject_lower = email["subject"].lower()
            body_lower = email["body"].lower()
            text = f"{subject_lower} {body_lower}"
            
            # Determine category
            email_category = "other"
            for category, keywords in self.email_categories.items():
                if any(keyword in text for keyword in keywords):
                    email_category = category
                    break
            
            # Determine priority
            priority = "normal"
            if any(urgent_word in text for urgent_word in ["urgent", "critical", "emergency", "asap"]):
                priority = "urgent"
            elif any(high_word in text for high_word in ["important", "priority", "deadline"]):
                priority = "high"
            
            email["category"] = email_category
            email["priority"] = priority
            categorized[email_category].append(email)
        
        # Store categorized emails in context
        context["categorized_emails"] = categorized
        
        # Generate category statistics
        stats = {category: len(emails) for category, emails in categorized.items() if len(emails) > 0}
        
        return {
            "success": True,
            "categorization_stats": stats,
            "urgent_count": sum(1 for email in emails if email.get("priority") == "urgent"),
            "high_priority_count": sum(1 for email in emails if email.get("priority") == "high")
        }
    
    def _process_urgent_emails(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process urgent emails first"""
        self.logger.info("Processing urgent emails")
        
        emails = context.get("emails", [])
        urgent_emails = [email for email in emails if email.get("priority") == "urgent"]
        
        processed_urgent = []
        actions_taken = []
        
        for email in urgent_emails:
            # Simulate urgent email processing
            action = {
                "email_id": email["id"],
                "subject": email["subject"],
                "action_taken": "escalated_to_admin",
                "timestamp": datetime.now().isoformat()
            }
            
            if parameters.get("auto_respond", False):
                # Generate automatic response
                response = self._generate_urgent_response(email)
                action["auto_response"] = response
                action["action_taken"] += ", auto_response_sent"
            
            processed_urgent.append(email["id"])
            actions_taken.append(action)
        
        context["processed_urgent"] = processed_urgent
        context["urgent_actions"] = actions_taken
        
        return {
            "success": True,
            "urgent_emails_processed": len(processed_urgent),
            "actions_taken": actions_taken
        }
    
    def _process_categorized_emails(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process emails by category"""
        self.logger.info("Processing categorized emails")
        
        categorized_emails = context.get("categorized_emails", {})
        processing_results = {}
        
        # Process meetings
        if parameters.get("process_meetings", True) and categorized_emails.get("meeting"):
            meeting_results = self._process_meeting_emails(categorized_emails["meeting"])
            processing_results["meetings"] = meeting_results
        
        # Process projects
        if parameters.get("process_projects", True) and categorized_emails.get("project"):
            project_results = self._process_project_emails(categorized_emails["project"])
            processing_results["projects"] = project_results
        
        # Auto-archive newsletters
        if parameters.get("auto_archive_newsletters", False) and categorized_emails.get("newsletter"):
            newsletter_results = self._archive_newsletters(categorized_emails["newsletter"])
            processing_results["newsletters"] = newsletter_results
        
        context["category_processing_results"] = processing_results
        
        return {
            "success": True,
            "processing_results": processing_results
        }
    
    def _generate_summary(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate email processing summary"""
        self.logger.info("Generating processing summary")
        
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_emails_processed": context.get("email_count", 0),
            "categorization_stats": context.get("categorized_emails", {}),
            "urgent_emails_handled": len(context.get("processed_urgent", [])),
            "actions_taken": context.get("urgent_actions", []),
            "category_results": context.get("category_processing_results", {})
        }
        
        if parameters.get("include_action_items", True):
            summary["action_items"] = self._extract_action_items(context)
        
        if parameters.get("include_statistics", True):
            summary["statistics"] = self._calculate_statistics(context)
        
        return {
            "success": True,
            "summary": summary
        }
    
    def _generate_urgent_response(self, email: Dict[str, Any]) -> str:
        """Generate automatic response for urgent emails"""
        return f"""
        Thank you for your urgent message regarding: {email['subject']}
        
        This email has been automatically flagged as urgent and forwarded to the appropriate team.
        We will respond as soon as possible.
        
        If this is a critical system issue, please also contact our emergency hotline.
        
        Best regards,
        AI-OS Email Agent
        """
    
    def _process_meeting_emails(self, meeting_emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process meeting-related emails"""
        processed = []
        for email in meeting_emails:
            # Extract meeting details (simplified)
            processed.append({
                "email_id": email["id"],
                "action": "calendar_integration_requested",
                "meeting_detected": True
            })
        
        return {
            "processed_count": len(processed),
            "actions": processed
        }
    
    def _process_project_emails(self, project_emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process project-related emails"""
        processed = []
        for email in project_emails:
            # Extract project information (simplified)
            processed.append({
                "email_id": email["id"],
                "action": "project_tracking_updated",
                "project_detected": True
            })
        
        return {
            "processed_count": len(processed),
            "actions": processed
        }
    
    def _archive_newsletters(self, newsletter_emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Archive newsletter emails"""
        archived = []
        for email in newsletter_emails:
            archived.append({
                "email_id": email["id"],
                "action": "archived",
                "reason": "newsletter_auto_archive"
            })
        
        return {
            "archived_count": len(archived),
            "actions": archived
        }
    
    def _extract_action_items(self, context: Dict[str, Any]) -> List[str]:
        """Extract action items from processing results"""
        action_items = []
        
        # Add urgent email follow-ups
        urgent_count = len(context.get("processed_urgent", []))
        if urgent_count > 0:
            action_items.append(f"Follow up on {urgent_count} urgent emails")
        
        # Add category-specific action items
        category_results = context.get("category_processing_results", {})
        if "meetings" in category_results:
            meeting_count = category_results["meetings"]["processed_count"]
            if meeting_count > 0:
                action_items.append(f"Review {meeting_count} meeting invitations")
        
        if "projects" in category_results:
            project_count = category_results["projects"]["processed_count"]
            if project_count > 0:
                action_items.append(f"Update project tracking for {project_count} items")
        
        return action_items
    
    def _calculate_statistics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate processing statistics"""
        total_emails = context.get("email_count", 0)
        categorized = context.get("categorized_emails", {})
        
        stats = {
            "total_processed": total_emails,
            "by_category": {cat: len(emails) for cat, emails in categorized.items() if len(emails) > 0},
            "urgent_emails": len(context.get("processed_urgent", [])),
            "processing_efficiency": "high" if total_emails > 0 else "n/a"
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Create agent instance
    agent = EmailProcessingAgent()
    
    # Test task planning
    print("=== Task Planning ===")
    task = "Check my unread emails and handle any urgent messages"
    plan = agent._plan_task(task)
    
    for step in plan:
        print(f"Step {step['step']}: {step['description']}")
    
    # Test execution
    print("\n=== Execution ===")
    result = agent.execute(task)
    
    print(f"Execution Status: {'Success' if result['success'] else 'Failed'}")
    print(f"Steps Completed: {result['steps_completed']}")
    
    if result.get('final_result'):
        summary = result['final_result'].get('summary', {})
        print(f"Emails Processed: {summary.get('total_emails_processed', 0)}")
        print(f"Urgent Emails: {summary.get('urgent_emails_handled', 0)}")
        
        if summary.get('action_items'):
            print("Action Items:")
            for item in summary['action_items']:
                print(f"  - {item}")