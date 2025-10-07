"""
Enhanced CLI for Sigma Agent with Prompt Toolkit
Features:
- Arrow key navigation through history
- Tab completion with command suggestions
- Multi-line input support
- Smart file drag & drop with auto-detection
- Real-time metrics display
- Advanced session manager
- Syntax highlighting
- Auto-suggestions from history

Usage:
    python sigma_cli.py
    python sigma_cli.py --new  # Start new session
"""

import asyncio
import os
import sys
import json
import mimetypes
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, PathCompleter, merge_completers
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Import from the original agent_Sigma module
import agent_Sigma

# Rich console for better output
console = Console()

# Workspace temp directory for CLI artifacts
TMP_DIR = Path(os.getenv("SIGMA_TMP_DIR", "tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Storage locations for history/session metadata
HISTORY_FILE = TMP_DIR / ".sigma_history"
SESSIONS_FILE = TMP_DIR / ".sigma_sessions.json"

# DeepSeek pricing (defaults assume cache-miss rates for DeepSeek-V3.2-Exp)
DEEPSEEK_INPUT_PRICE_PER_TOKEN = float(os.getenv("SIGMA_DEEPSEEK_INPUT_PRICE", "0.28")) / 1_000_000
DEEPSEEK_OUTPUT_PRICE_PER_TOKEN = float(os.getenv("SIGMA_DEEPSEEK_OUTPUT_PRICE", "0.42")) / 1_000_000

# Custom style for the prompt
prompt_style = Style.from_dict({
    'prompt': '#00aa00 bold',
    'user-input': '#ffffff',
})

# Command suggestions for tab completion
COMMANDS = [
    'exit', 'quit', 'bye', 'clear', 'help',
    'history', 'sessions', 'new', 'status',
    '/file', '/search', '/math', '/shell',
    '/metrics', '/session', '/list'
]

# Global metrics tracker
class MetricsTracker:
    def __init__(self):
        self.start_time = None
        self.tokens_input = 0
        self.tokens_output = 0
        self.total_tokens = 0
        self.response_time = 0
        self.tools_called = []
        
    def start(self):
        self.start_time = time.time()
        self.tokens_input = 0
        self.tokens_output = 0
        self.tools_called = []
        
    def stop(self):
        if self.start_time:
            self.response_time = time.time() - self.start_time
            
    def update_tokens(self, input_tokens=0, output_tokens=0):
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.total_tokens = self.tokens_input + self.tokens_output
        
    def add_tool(self, tool_name: str):
        self.tools_called.append(tool_name)

metrics = MetricsTracker()


async def prompt_for_session_id() -> Optional[str]:
    """Prompt the user for a session identifier without blocking."""
    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=create_completer(),
        style=prompt_style,
        bottom_toolbar=get_bottom_toolbar,
        multiline=False,
        enable_history_search=True,
    )

    try:
        value = await session.prompt_async(
            HTML('<prompt>Session ID:</prompt> '),
            rprompt=HTML('<ansigray>[session]</ansigray>')
        )
    except (KeyboardInterrupt, EOFError):
        return None

    value = value.strip()
    return value or None


def load_sessions() -> Dict:
    """Load session metadata from JSON file"""
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_session_metadata(session_id: str, title: str = None, tags: List[str] = None):
    """Save session metadata"""
    sessions = load_sessions()
    
    if session_id not in sessions:
        sessions[session_id] = {
            'created': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
            'title': title or f"Session {session_id[:8]}",
            'tags': tags or [],
            'message_count': 0
        }
    else:
        sessions[session_id]['last_used'] = datetime.now().isoformat()
        sessions[session_id]['message_count'] = sessions[session_id].get('message_count', 0) + 1
        
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

def detect_file_type(file_path: Path) -> str:
    """Detect file type and return appropriate handler"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    suffix = file_path.suffix.lower()
    
    # Code files
    code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.rb', '.php'}
    if suffix in code_extensions:
        return 'code'
    
    # Data files
    if suffix in {'.csv', '.json', '.xml', '.yaml', '.yml'}:
        return 'data'
    
    # Documents
    if suffix in {'.txt', '.md', '.pdf', '.docx', '.doc'}:
        return 'document'
    
    # Images
    if mime_type and mime_type.startswith('image/'):
        return 'image'
    
    return 'unknown'

def create_file_preview(file_path: Path, max_lines: int = 10) -> str:
    """Create a preview of file content"""
    try:
        if file_path.stat().st_size > 1024 * 1024:  # 1MB
            return f"[Large file: {file_path.stat().st_size / 1024 / 1024:.2f} MB]"
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
            preview = ''.join(lines)
            if len(f.readlines()) > max_lines:
                preview += f"\n... ({len(f.readlines()) - max_lines} more lines)"
            return preview
    except:
        return "[Binary file - cannot preview]"

def create_completer():
    """Create a completer with command and path completion"""
    command_completer = WordCompleter(
        COMMANDS,
        ignore_case=True,
        sentence=True,
        match_middle=True
    )
    path_completer = PathCompleter(expanduser=True)
    return merge_completers([command_completer, path_completer])

def get_bottom_toolbar():
    """Display helpful shortcuts at the bottom"""
    return HTML(
        ' <b>Shortcuts:</b> '
        '<style bg="ansidarkgray"> Ctrl+D </style> Exit | '
        '<style bg="ansidarkgray"> Ctrl+C </style> Cancel | '
        '<style bg="ansidarkgray"> Tab </style> Complete | '
        '<style bg="ansidarkgray"> ↑↓ </style> History | '
        '<style bg="ansidarkgray"> /sessions </style> Manager'
    )

def print_welcome():
    """Print welcome message with available commands"""
    welcome_text = """
# 🌌 Sigma Agent - Enhanced CLI v2.0

**Core Commands:**
- `exit`, `quit`, `bye` - Exit the application
- `clear` - Clear the screen
- `help` - Show this help message
- `status` - Show agent status
- `metrics` - Show detailed metrics

**Session Management:**
- `sessions` or `/sessions` - Open session manager
- `new` - Start a new session
- `history` - Show command history

**Special Prefixes:**
- `/file <path>` - Smart file analysis (auto-detects type)
- `/search <query>` - Web search
- `/math <expression>` - Calculate
- `/shell <command>` - Execute shell command (with confirmation)

**Tips:**
- Use ↑/↓ arrows to navigate history
- Press Tab for autocompletion
- **Drag & drop files** - automatic type detection!
- Multi-file support: `/file file1.py file2.csv`
"""
    console.print(Markdown(welcome_text))
    console.print("─" * console.width)

def show_metrics_panel():
    """Display current session metrics"""
    table = Table(title="📊 Session Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")
    
    table.add_row("Response Time", f"{metrics.response_time:.2f}s")
    table.add_row("Input Tokens", str(metrics.tokens_input))
    table.add_row("Output Tokens", str(metrics.tokens_output))
    table.add_row("Total Tokens", str(metrics.total_tokens))
    
    if metrics.tools_called:
        table.add_row("Tools Called", ", ".join(metrics.tools_called))
    
    # Estimate cost using DeepSeek cache-miss pricing (override via SIGMA_DEEPSEEK_* env vars)
    estimated_cost = (
        metrics.tokens_input * DEEPSEEK_INPUT_PRICE_PER_TOKEN
        + metrics.tokens_output * DEEPSEEK_OUTPUT_PRICE_PER_TOKEN
    )
    table.add_row("Estimated Cost", f"${estimated_cost:.6f}")

    console.print(table)
    console.print("[dim]Assuming DeepSeek cache-miss rates; set SIGMA_DEEPSEEK_INPUT_PRICE / SIGMA_DEEPSEEK_OUTPUT_PRICE to override.[/dim]")

async def show_session_manager(session: PromptSession) -> Optional[str]:
    """Display interactive session manager"""
    sessions = load_sessions()

    if not sessions:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return None
    
    table = Table(title="💾 Session Manager", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Session ID", style="cyan", width=12)
    table.add_column("Title", style="white", width=25)
    table.add_column("Messages", justify="right", width=10)
    table.add_column("Last Used", style="green", width=20)
    table.add_column("Tags", style="magenta")
    
    session_list = []
    for idx, (session_id, data) in enumerate(sorted(
        sessions.items(), 
        key=lambda x: x[1].get('last_used', ''), 
        reverse=True
    ), 1):
        last_used = datetime.fromisoformat(data['last_used'])
        time_ago = (datetime.now() - last_used).total_seconds()
        
        if time_ago < 3600:
            time_str = f"{int(time_ago / 60)}m ago"
        elif time_ago < 86400:
            time_str = f"{int(time_ago / 3600)}h ago"
        else:
            time_str = f"{int(time_ago / 86400)}d ago"
        
        table.add_row(
            str(idx),
            session_id[:8] + "...",
            data.get('title', 'Untitled'),
            str(data.get('message_count', 0)),
            time_str,
            ", ".join(data.get('tags', [])[:2])
        )
        session_list.append(session_id)
    
    console.print(table)
    console.print("\n[dim]Enter session number to switch, or press Enter to cancel[/dim]")

    try:
        choice = await session.prompt_async(
            HTML('<prompt>Select session:</prompt> '),
            rprompt=HTML('<ansigray>[sessions]</ansigray>'),
        )
    except (KeyboardInterrupt, EOFError):
        return None

    choice = choice.strip()
    if choice and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(session_list):
            return session_list[idx]

    return None

async def handle_special_commands(user_input: str, session: PromptSession, current_session_id: str) -> Optional[str]:
    """Handle special CLI commands"""
    cmd = user_input.lower().strip()

    if cmd in ['exit', 'quit', 'bye']:
        console.print("\n[bold cyan]👋 Goodbye![/bold cyan]")
        sys.exit(0)
    
    elif cmd == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        print_welcome()
        return None
    
    elif cmd == 'help':
        print_welcome()
        return None
    
    elif cmd == 'metrics':
        show_metrics_panel()
        return None

    elif cmd in ['sessions', '/sessions']:
        new_session = await show_session_manager(session)
        if new_session:
            console.print(f"\n[green]✓[/green] Switching to session: {new_session[:8]}...")
            return f"SWITCH_SESSION:{new_session}"
        return None
    
    elif cmd == 'history':
        console.print("\n[bold]Command History:[/bold]")
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                lines = f.readlines()[-20:]
                for i, line in enumerate(lines, 1):
                    console.print(f"{i:3d}. {line.strip()}")
        else:
            console.print("[dim]No history available yet[/dim]")
        return None
    
    elif cmd == 'new':
        console.print("\n[bold yellow]⚠️  Starting new session...[/bold yellow]")
        return "NEW_SESSION"
    
    elif cmd == 'status':
        console.print("\n[bold]Agent Status:[/bold]")
        console.print(f"Session ID: {current_session_id}")
        console.print(f"History file: {HISTORY_FILE}")
        console.print(f"Sessions file: {SESSIONS_FILE}")
        show_metrics_panel()
        return None
    
    return user_input

def process_smart_file(file_path_str: str) -> Optional[str]:
    """Smart file processing with type detection"""
    file_path = Path(file_path_str.strip().strip('"').strip("'"))
    
    if not file_path.exists():
        console.print(f"[red]✗ File not found: {file_path}[/red]")
        return None
    
    if not file_path.is_file():
        console.print(f"[red]✗ Not a file: {file_path}[/red]")
        return None
    
    # Detect file type
    file_type = detect_file_type(file_path)
    file_size = file_path.stat().st_size / 1024  # KB
    
    # Create info panel
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column(style="cyan bold")
    info_table.add_column(style="white")
    
    info_table.add_row("📄 File:", file_path.name)
    info_table.add_row("📦 Type:", file_type.upper())
    info_table.add_row("💾 Size:", f"{file_size:.2f} KB")
    info_table.add_row("📍 Path:", str(file_path.absolute()))
    
    console.print(Panel(info_table, title="[bold green]✓ File Detected[/bold green]", border_style="green"))
    
    # Show preview
    if file_type in ['code', 'document', 'data']:
        preview = create_file_preview(file_path)
        if preview:
            console.print("\n[bold]Preview:[/bold]")
            console.print(Panel(preview[:500], border_style="dim"))
    
    # Read content
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Create context-aware prompt based on file type
        prompts = {
            'code': f"Analyze this {file_path.suffix} code file ({file_path.name}):\n\n{content}",
            'data': f"Analyze this data file ({file_path.name}):\n\n{content}",
            'document': f"Analyze this document ({file_path.name}):\n\n{content}",
            'image': f"This is an image file: {file_path.name}",
            'unknown': f"Analyze this file ({file_path.name}):\n\n{content}"
        }
        
        return prompts.get(file_type, prompts['unknown'])
        
    except Exception as e:
        console.print(f"[red]✗ Error reading file: {e}[/red]")
        return None

def process_input(user_input: str) -> Optional[str]:
    """Process user input and handle special prefixes"""
    
    # Handle /file prefix with smart detection
    if user_input.startswith('/file '):
        file_paths = user_input[6:].strip().split()
        
        if len(file_paths) == 1:
            return process_smart_file(file_paths[0])
        else:
            # Multi-file support
            all_content = []
            for fp in file_paths:
                content = process_smart_file(fp)
                if content:
                    all_content.append(content)
            
            if all_content:
                return "\n\n---\n\n".join(all_content)
            return None
    
    # Auto-detect if input is a file path (drag & drop)
    stripped = user_input.strip().strip('"').strip("'")
    if Path(stripped).exists() and Path(stripped).is_file():
        console.print("\n[yellow]🔍 File detected! Processing...[/yellow]")
        return process_smart_file(stripped)
    
    return user_input

async def run_enhanced_repl(agent_instance):
    """Enhanced REPL with all features"""
    
    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=create_completer(),
        complete_while_typing=True,
        bottom_toolbar=get_bottom_toolbar,
        style=prompt_style,
        multiline=False,
        enable_history_search=True,
    )
    
    session._sigma_session_id = agent_instance.session_id
    current_session_id = agent_instance.session_id
    
    print_welcome()
    console.print(f"[bold green]🤖 Sigma Agent ready![/bold green] Session: [cyan]{current_session_id[:8]}...[/cyan]\n")
    
    # Save initial session metadata
    save_session_metadata(current_session_id)
    
    while True:
        try:
            user_input = await session.prompt_async(
                HTML('<prompt>You:</prompt> '),
                rprompt=HTML(f'<ansigray>[{current_session_id[:8]}...]</ansigray>')
            )
            
            if not user_input.strip():
                continue
            
            # Handle special commands
            processed = await handle_special_commands(user_input, session, current_session_id)
            
            if processed is None:
                continue
            
            if processed == "NEW_SESSION":
                return "NEW_SESSION"
            
            if processed.startswith("SWITCH_SESSION:"):
                new_session = processed.split(":")[1]
                return f"SWITCH_SESSION:{new_session}"
            
            # Process input (file detection, etc)
            processed = process_input(processed)
            
            if processed is None:
                continue
            
            # Start metrics tracking
            metrics.start()
            
            # Display agent response with live metrics
            console.print("\n[bold cyan]Assistant:[/bold cyan]")
            
            try:
                # Create a progress indicator
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False
                ) as progress:
                    task = progress.add_task("[cyan]Generating response...", total=None)
                    
                    # Run the agent
                    agent_instance.print_response(processed, stream=True)
                    progress.update(task, completed=True)
                
                # Stop metrics and update from agent response
                metrics.stop()
                
                # Try to extract metrics from agent response
                if hasattr(agent_instance, 'run_response') and agent_instance.run_response:
                    resp_metrics = getattr(agent_instance.run_response, 'metrics', {})
                    if resp_metrics:
                        input_tok = resp_metrics.get('input_tokens', [0])
                        output_tok = resp_metrics.get('output_tokens', [0])
                        metrics.update_tokens(
                            sum(input_tok) if isinstance(input_tok, list) else input_tok,
                            sum(output_tok) if isinstance(output_tok, list) else output_tok
                        )
                
                # Show metrics inline
                console.print(f"\n[dim]⏱️  {metrics.response_time:.2f}s | "
                            f"📊 {metrics.total_tokens} tokens[/dim]")
                
                # Update session metadata
                save_session_metadata(current_session_id)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Generation interrupted.[/yellow]\n")
                continue
            
            except Exception as exc:
                console.print(f"\n[red]✗ Error: {exc}[/red]\n")
                if agent_Sigma.ENABLE_DEBUG_MODE:
                    raise
                continue
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' or Ctrl+D to quit.[/yellow]")
            continue
        
        except EOFError:
            console.print("\n[bold cyan]👋 Goodbye![/bold cyan]")
            break

def main():
    """Main entry point"""
    agent_Sigma.register_session_prompt_resolver(prompt_for_session_id)
    current_session_id = None

    try:
        while True:
            if current_session_id:
                # Use existing session
                resolved_session = current_session_id
            else:
                # New session
                session_id_choice = agent_Sigma.resolve_session_id_from_user()
                resolved_session = session_id_choice or agent_Sigma.default_session_id()

            agent_instance = asyncio.run(agent_Sigma.initialize_agent(session_id=resolved_session))

            status = getattr(agent_instance, "bootstrap_status", None)
            if status:
                console.print("[dim]Startup status: "
                              f"knowledge={'ok' if status.knowledge_loaded else 'skip'}"
                              f" | model={'ok' if status.model_loaded else 'error'}"
                              f" | reasoning={'ok' if status.reasoning_model_loaded else 'skip'}[/dim]")
                if status.knowledge_error:
                    console.print(f"[yellow]Knowledge load warning:[/yellow] {status.knowledge_error}")
                if status.model_error:
                    console.print(f"[red]Model error:[/red] {status.model_error}")
                if status.reasoning_model_error:
                    console.print(f"[yellow]Reasoning model warning:[/yellow] {status.reasoning_model_error}")

            result = asyncio.run(run_enhanced_repl(agent_instance))

            if result == "NEW_SESSION":
                current_session_id = None
                continue
            elif result and result.startswith("SWITCH_SESSION:"):
                current_session_id = result.split(":")[1]
                continue
            else:
                break
    finally:
        agent_Sigma.register_session_prompt_resolver(None)

if __name__ == "__main__":
    main()
