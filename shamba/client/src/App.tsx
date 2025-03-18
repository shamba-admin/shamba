import "./App.css";
import { Button } from "@/components/ui/button";

function App() {
	return (
		<div className="min-h-screen flex flex-col">
			{/* Top section */}
			<header className="w-full bg-primary text-primary-foreground p-4">
				<h1 className="text-2xl font-bold">Header</h1>
			</header>

			<div className="flex flex-1">
				{/* Side menu */}
				<nav className="w-1/4 max-w-xs bg-secondary p-4">
					<ul className="space-y-2">
						<li>
							<Button variant="ghost" className="w-full justify-start">
								Menu Item 1
							</Button>
						</li>
						<li>
							<Button variant="ghost" className="w-full justify-start">
								Menu Item 2
							</Button>
						</li>
						<li>
							<Button variant="ghost" className="w-full justify-start">
								Menu Item 3
							</Button>
						</li>
					</ul>
				</nav>

				{/* Main content area */}
				<main className="flex-1 p-4">Hello</main>
			</div>
		</div>
	);
}

export default App;
