
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const todoInput = document.getElementById('todo-input');
    const todoPriority = document.getElementById('todo-priority');
    const addTodoBtn = document.getElementById('add-todo-btn');
    const todoList = document.getElementById('todo-list');
    const tasksLeft = document.getElementById('tasks-left');
    const clearCompletedBtn = document.getElementById('clear-completed');
    const filterBtns = document.querySelectorAll('.filter-btn');

    // Load todos from localStorage
    let todos = JSON.parse(localStorage.getItem('mlprompt-todos') || '[]');
    let filter = 'all';

    // Render initial todos
    renderTodos();

    // Event listeners
    addTodoBtn.addEventListener('click', addTodo);
    todoInput.addEventListener('keyup', function(e) {
        if (e.key === 'Enter') addTodo();
    });
    clearCompletedBtn.addEventListener('click', clearCompleted);
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            filter = this.dataset.filter;
            renderTodos();
        });
    });

    // Functions
    function addTodo() {
        const text = todoInput.value.trim();
        if (text === '') return;

        const todo = {
            id: Date.now(),
            text: text,
            priority: todoPriority.value,
            completed: false,
            createdAt: new Date().toISOString()
        };

        todos.push(todo);
        saveTodos();
        todoInput.value = '';
        renderTodos();
    }

    function toggleTodo(id) {
        todos = todos.map(todo => {
            if (todo.id === id) {
                todo.completed = !todo.completed;
                
                // If this is a task being completed, update the TODO.md file via API
                if (todo.completed) {
                    updateTodoMarkdown(todo.text, true);
                } else {
                    updateTodoMarkdown(todo.text, false);
                }
            }
            return todo;
        });
        saveTodos();
        renderTodos();
    }

    function deleteTodo(id) {
        todos = todos.filter(todo => todo.id !== id);
        saveTodos();
        renderTodos();
    }

    function clearCompleted() {
        todos = todos.filter(todo => !todo.completed);
        saveTodos();
        renderTodos();
    }

    function renderTodos() {
        // Filter todos based on current filter
        let filteredTodos = todos;
        if (filter === 'active') {
            filteredTodos = todos.filter(todo => !todo.completed);
        } else if (filter === 'completed') {
            filteredTodos = todos.filter(todo => todo.completed);
        }

        // Sort by priority and creation date
        filteredTodos.sort((a, b) => {
            const priorityOrder = { high: 1, medium: 2, low: 3 };
            if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
                return priorityOrder[a.priority] - priorityOrder[b.priority];
            }
            return new Date(b.createdAt) - new Date(a.createdAt);
        });

        // Clear the list
        todoList.innerHTML = '';

        // Add todos to the list
        filteredTodos.forEach(todo => {
            const li = document.createElement('li');
            li.className = `todo-item priority-${todo.priority}`;
            if (todo.completed) li.classList.add('completed');

            li.innerHTML = `
                <input type="checkbox" ${todo.completed ? 'checked' : ''}>
                <span class="todo-text">${todo.text}</span>
                <span class="todo-priority-badge">${todo.priority}</span>
                <button class="delete-btn"><i class="fas fa-trash"></i></button>
            `;

            // Add event listeners
            li.querySelector('input[type="checkbox"]').addEventListener('change', () => toggleTodo(todo.id));
            li.querySelector('.delete-btn').addEventListener('click', () => deleteTodo(todo.id));

            todoList.appendChild(li);
        });

        // Update tasks left count
        const activeTodos = todos.filter(todo => !todo.completed);
        tasksLeft.textContent = `${activeTodos.length} task${activeTodos.length !== 1 ? 's' : ''} left`;
        
        // Update progress indicators
        updateProgressIndicators();
    }

    function updateProgressIndicators() {
        // Calculate completion percentages by priority
        const highPriorityTasks = todos.filter(todo => todo.priority === 'high');
        const mediumPriorityTasks = todos.filter(todo => todo.priority === 'medium');
        const lowPriorityTasks = todos.filter(todo => todo.priority === 'low');
        
        const highCompleted = highPriorityTasks.filter(todo => todo.completed).length;
        const mediumCompleted = mediumPriorityTasks.filter(todo => todo.completed).length;
        const lowCompleted = lowPriorityTasks.filter(todo => todo.completed).length;
        
        const highPercentage = highPriorityTasks.length ? Math.round((highCompleted / highPriorityTasks.length) * 100) : 0;
        const mediumPercentage = mediumPriorityTasks.length ? Math.round((mediumCompleted / mediumPriorityTasks.length) * 100) : 0;
        const lowPercentage = lowPriorityTasks.length ? Math.round((lowCompleted / lowPriorityTasks.length) * 100) : 0;
        const totalPercentage = todos.length ? Math.round((todos.filter(todo => todo.completed).length / todos.length) * 100) : 0;
        
        // If we have an existing progress container, remove it
        const existingProgress = document.querySelector('.todo-progress-container');
        if (existingProgress) {
            existingProgress.remove();
        }
        
        // Create progress indicators
        const progressContainer = document.createElement('div');
        progressContainer.className = 'todo-progress-container';
        
        progressContainer.innerHTML = `
            <h3>Project Progress</h3>
            <div class="progress-item">
                <div class="progress-label">Overall: ${totalPercentage}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${totalPercentage}%"></div>
                </div>
            </div>
            <div class="progress-item">
                <div class="progress-label">High Priority: ${highPercentage}%</div>
                <div class="progress-bar">
                    <div class="progress-fill high" style="width: ${highPercentage}%"></div>
                </div>
            </div>
            <div class="progress-item">
                <div class="progress-label">Medium Priority: ${mediumPercentage}%</div>
                <div class="progress-bar">
                    <div class="progress-fill medium" style="width: ${mediumPercentage}%"></div>
                </div>
            </div>
            <div class="progress-item">
                <div class="progress-label">Low Priority: ${lowPercentage}%</div>
                <div class="progress-bar">
                    <div class="progress-fill low" style="width: ${lowPercentage}%"></div>
                </div>
            </div>
        `;
        
        // Insert before the todo list
        todoList.parentNode.insertBefore(progressContainer, todoList);
    }

    // Optional: Sync changes to TODO.md file via API
    async function updateTodoMarkdown(taskText, isCompleted) {
        try {
            // This would call an API endpoint to update the TODO.md file
            // In a real implementation, you'd have an endpoint to handle this
            console.log(`Task "${taskText}" marked as ${isCompleted ? 'completed' : 'incomplete'}`);
            
            // Mock API call for demonstration
            /*
            await fetch('/api/update_todo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    task: taskText,
                    completed: isCompleted
                }),
            });
            */
        } catch (error) {
            console.error('Error updating TODO.md:', error);
        }
    }

    function saveTodos() {
        localStorage.setItem('mlprompt-todos', JSON.stringify(todos));
    }
});
